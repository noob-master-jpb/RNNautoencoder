import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
from model import test_model  # Imports your architecture
import polars as pd
import random
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")

# --- 1. Settings & Initialization ---
INPUT_DIM = 27
HIDDEN_DIM = 8
EOS_IDX = 26  # The index for the space " " character
BATCH_SIZE = 5000
LEARNING_RATE = 0.01
EPOCHS = 500
VAL_SPLIT = 0.1
USE_BEST = True
BEST_PATH = "best.pt"
LR_FACTOR = 0.5
LR_PATIENCE = 10
LR_MIN = 1e-6

# Initialize the model, optimizer, and loss function
model = test_model(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, eos_index=EOS_IDX)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=LR_MIN,
)
best_val_loss = float("inf")

if USE_BEST:
    if os.path.exists(BEST_PATH):
        try:
            state = torch.load(BEST_PATH, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded best model from {BEST_PATH}")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Could not load {BEST_PATH}: {exc}. Training from scratch.")
    else:
        print(f"{BEST_PATH} not found. Training from scratch.")


class WordDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class LengthBucketBatchSampler(Sampler):
    def __init__(self, buckets, batch_size, shuffle=True, drop_last=False):
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        bucket_keys = list(self.buckets.keys())
        if self.shuffle:
            random.shuffle(bucket_keys)

        for key in bucket_keys:
            indices = list(self.buckets[key])
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def collate_same_length(batch):
    return torch.stack(batch, dim=0)

# --- 2. Data Preparation (Bucketing) ---
print("Loading data and building buckets...")
df = pd.read_parquet("words.parquet")

alphabet = "abcdefghijklmnopqrstuvwxyz "
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}

sequences = []
seq_lengths = []
# Group words strictly by length
for word in df["words"]:
    word_with_eof = f"{word} "
    if any(char not in char_to_idx for char in word_with_eof):
        continue

    indices = [char_to_idx[char] for char in word_with_eof]
    seq_tensor = torch.tensor(indices, dtype=torch.long)
    sequences.append(seq_tensor)
    seq_lengths.append(len(seq_tensor))

all_indices = list(range(len(sequences)))
random.shuffle(all_indices)
val_size = int(len(all_indices) * VAL_SPLIT)
if len(all_indices) > 1:
    val_size = max(1, val_size)

val_index_set = set(all_indices[:val_size])
train_buckets = defaultdict(list)
val_buckets = defaultdict(list)
for idx in all_indices:
    length = seq_lengths[idx]
    if idx in val_index_set:
        val_buckets[length].append(idx)
    else:
        train_buckets[length].append(idx)

# --- 3. Batch Generation ---
dataset = WordDataset(sequences)
print(len(train_buckets.get(8, [])))  # Print available sequence lengths for verification
num_workers = 2 if device.type == "cuda" else 0
pin_memory = device.type == "cuda"
train_batch_sampler = LengthBucketBatchSampler(train_buckets, BATCH_SIZE, shuffle=True, drop_last=False)
val_batch_sampler = LengthBucketBatchSampler(val_buckets, BATCH_SIZE, shuffle=False, drop_last=False)
loader_kwargs = {
    "collate_fn": collate_same_length,
    "num_workers": num_workers,
    "pin_memory": pin_memory,
    "persistent_workers": num_workers > 0,
}
if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
training_loader = DataLoader(dataset, batch_sampler=train_batch_sampler, **loader_kwargs)
val_loader = DataLoader(dataset, batch_sampler=val_batch_sampler, **loader_kwargs)

print(f"Total uniform batches created: {len(training_loader)}\n")

# --- 4. The Training Loop ---
print("Starting Training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, batch_indices in enumerate(training_loader):
        # batch_indices shape: [Batch_Size, Dynamic_Length]
        batch_indices = batch_indices.to(device, non_blocking=True)
        batch_tensor = F.one_hot(batch_indices, num_classes=INPUT_DIM).to(dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        # 1. Get dynamic sequence length for this specific batch
        current_seq_length = batch_tensor.size(1)

        # 2. Forward Pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            predictions = model(batch_tensor, max_length=current_seq_length, stop_on_eos=False)

            # 3. Calculate Loss
            # Flatten tensors to satisfy CrossEntropyLoss requirements:
            # Predictions: [Batch * Seq_Len, 27]
            # Targets: [Batch * Seq_Len]
            loss = loss_function(predictions.view(-1, INPUT_DIM), batch_indices.view(-1))

        # 4. Backpropagate & Update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(training_loader)
    model.eval()
    total_val_loss = 0
    val_steps = 0
    with torch.no_grad():
        for batch_indices in val_loader:
            batch_indices = batch_indices.to(device, non_blocking=True)
            batch_tensor = F.one_hot(batch_indices, num_classes=INPUT_DIM).to(dtype=torch.float32)
            current_seq_length = batch_tensor.size(1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(batch_tensor, max_length=current_seq_length, stop_on_eos=False)
                val_loss = loss_function(predictions.view(-1, INPUT_DIM), batch_indices.view(-1))

            total_val_loss += val_loss.item()
            val_steps += 1

    if val_steps > 0:
        avg_val_loss = total_val_loss / val_steps
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), BEST_PATH)
                print(f"Saved best model to {BEST_PATH}")
            except OSError as exc:
                print(f"Could not save {BEST_PATH}: {exc}")
        val_loss_str = f"{avg_val_loss:.4f}"
    else:
        val_loss_str = "N/A"

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f} - "
        f"Val Loss: {val_loss_str} - LR: {current_lr:.8f}"
    )

print("\nTraining Complete!")   
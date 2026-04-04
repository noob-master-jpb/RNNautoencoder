import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
from model import test_model
import polars as pd
import random
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")

INPUT_DIM = 27
HIDDEN_DIM = 64
EOS_IDX = 26
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
EPOCHS = 500
VAL_SPLIT = 0.1
USE_BEST = False
BEST_PATH = "best.pt"
LR_FACTOR = 0.9
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
best_val_acc = 0.0

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

print("Loading data and building buckets...")
df = pd.read_parquet("words.parquet")

alphabet = "abcdefghijklmnopqrstuvwxyz "
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}

sequences = []
seq_lengths = []
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


dataset = WordDataset(sequences)
print(len(train_buckets.get(8, [])))
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

print("Starting Training...")

for epoch in range(EPOCHS):
    teacher_forcing_ratio = max(0.1, 0.8 * (0.98 ** epoch))
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, batch_indices in enumerate(training_loader):
        batch_indices = batch_indices.to(device, non_blocking=True)
        batch_tensor = F.one_hot(batch_indices, num_classes=INPUT_DIM).to(dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        current_seq_length = batch_tensor.size(1)
        with torch.amp.autocast("cuda", enabled=use_amp):
            predictions = model(batch_tensor, max_length=current_seq_length, stop_on_eos=False,teacher_forcing_ratio=teacher_forcing_ratio)
            loss = loss_function(predictions.view(-1, INPUT_DIM), batch_indices.view(-1))

        pred_indices = predictions.detach().argmax(dim=-1)
        train_correct += (pred_indices == batch_indices).sum().item()
        train_total += batch_indices.numel()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(training_loader)
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    model.eval()
    total_val_loss = 0
    val_steps = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_indices in val_loader:
            batch_indices = batch_indices.to(device, non_blocking=True)
            batch_tensor = F.one_hot(batch_indices, num_classes=INPUT_DIM).to(dtype=torch.float32)
            current_seq_length = batch_tensor.size(1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(batch_tensor, max_length=current_seq_length, stop_on_eos=False)
                val_loss = loss_function(predictions.view(-1, INPUT_DIM), batch_indices.view(-1))

            pred_indices = predictions.argmax(dim=-1)
            val_correct += (pred_indices == batch_indices).sum().item()
            val_total += batch_indices.numel()
            total_val_loss += val_loss.item()
            val_steps += 1

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
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    val_acc_str = f"{val_acc:.4f}"

  

    train_acc_str = f"{train_acc:.4f}"

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f} - "
        f"Train Acc: {train_acc_str} - Val Loss: {val_loss_str} - "
        f"Val Acc: {val_acc_str} - Best Val Acc: {best_val_acc:.4f} - "
        f"LR: {current_lr:.8f}"
    )


print("\nTraining Complete!")   
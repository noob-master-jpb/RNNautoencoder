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
HIDDEN_DIM = 440
EOS_IDX = 26
BATCH_SIZE = 512*14
LEARNING_RATE = 0.001
EPOCHS = 500

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.5

TF_START = 1.0
TF_END = 0.2
TF_DECAY_EPOCHS = 120

KL_MAX = 0.05
KL_WARMUP_EPOCHS = 80
KL_FREE_BITS = 0.02

VAL_SPLIT = 0.1
USE_BEST = True
BEST_PATH = "best.pt"
LR_FACTOR = 0.5
LR_PATIENCE = 5
LR_MIN = 1e-10

# Initialize the model, optimizer, and loss function
model = test_model(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, eos_index=EOS_IDX)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(
                    f"Loaded {BEST_PATH} with incompatible keys. "
                    f"Missing: {len(incompatible.missing_keys)}, "
                    f"Unexpected: {len(incompatible.unexpected_keys)}"
                )
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


def kl_weight(epoch):
    if KL_WARMUP_EPOCHS <= 0:
        return KL_MAX
    return min(KL_MAX, KL_MAX * (epoch + 1) / KL_WARMUP_EPOCHS)


def teacher_forcing_schedule(epoch):
    if TF_DECAY_EPOCHS <= 0:
        return TF_END
    progress = min(1.0, epoch / TF_DECAY_EPOCHS)
    return TF_START + (TF_END - TF_START) * progress


def compute_kl(mu, logvar):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if KL_FREE_BITS > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=KL_FREE_BITS)
    return kl_per_dim.sum(dim=1).mean()

print("Loading data and building buckets...")
df_words = pd.read_parquet("words.parquet")
df_train_words = pd.read_parquet("train_words.parquet")
df = pd.concat([df_words, df_train_words], how="vertical")

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

num_workers = 6 if device.type == "cuda" else 0
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
    loader_kwargs["prefetch_factor"] = 1
training_loader = DataLoader(dataset, batch_sampler=train_batch_sampler, **loader_kwargs)
val_loader = DataLoader(dataset, batch_sampler=val_batch_sampler, **loader_kwargs)

print(f"Total uniform batches created: {len(training_loader)}\n")

print("Starting Training...")

for epoch in range(EPOCHS):
    teacher_forcing_ratio = teacher_forcing_schedule(epoch)
    beta = kl_weight(epoch)

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
            # --- model now returns mu, logvar ---
            predictions, mu, logvar = model(
                batch_tensor,
                max_length=current_seq_length,
                stop_on_eos=False,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            # --- reconstruction loss (unchanged) ---
            recon_loss = loss_function(
                predictions.view(-1, INPUT_DIM),
                batch_indices.view(-1)
            )

            # --- KL loss (added) ---
            kl_loss = compute_kl(mu, logvar)

            # --- total loss ---
            loss = recon_loss + beta * kl_loss

        pred_indices = predictions.detach().argmax(dim=-1)
        train_correct += (pred_indices == batch_indices).sum().item()
        train_total += batch_indices.numel()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(training_loader)
    train_acc = train_correct / train_total if train_total > 0 else 0.0

    # ---------------- VALIDATION ----------------

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
                predictions, mu, logvar = model(
                    batch_tensor,
                    max_length=current_seq_length,
                    stop_on_eos=False
                )

                recon_loss = loss_function(
                    predictions.view(-1, INPUT_DIM),
                    batch_indices.view(-1)
                )

                kl_loss = compute_kl(mu, logvar)

                val_loss = recon_loss + beta * kl_loss

            pred_indices = predictions.argmax(dim=-1)
            val_correct += (pred_indices == batch_indices).sum().item()
            val_total += batch_indices.numel()
            total_val_loss += val_loss.item()
            val_steps += 1

    if val_steps > 0:
        avg_val_loss = total_val_loss / val_steps
        scheduler.step(avg_val_loss)
    else:
        avg_val_loss = float("inf")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        try:
            torch.save(model.state_dict(), BEST_PATH)
            print(f"Saved best model to {BEST_PATH}")
        except OSError as exc:
            print(f"Could not save {BEST_PATH}: {exc}")

    val_loss_str = f"{avg_val_loss:.3f}"
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    val_acc_str = f"{val_acc:.3f}"
    train_acc_str = f"{train_acc:.3f}"

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] - "
        f"Train:- Loss: {avg_loss:.3f} - Acc: {train_acc_str} -     "
        f"Val:- Loss: {val_loss_str} - Acc: {val_acc_str} -      "
        f"Best Val Acc: {best_val_acc:.3f} -    "
        f"LR: {current_lr:.8f}"
    )

print("\nTraining Complete!")
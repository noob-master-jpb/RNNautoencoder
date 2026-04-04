import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. The Encoder ---
class RNNencoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=8):
        super().__init__()
        self.active = nn.Tanh()
        self.model = nn.Sequential(
            nn.Linear(input_size + hidden_size, 32),
            self.active,
            nn.Linear(32, 16),
            self.active,
            nn.Linear(16, hidden_size),
            self.active # Bounds final memory
        )
        
    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        return self.model(combined)

# --- 2. The Decoder ---
    class RNNdecoder(nn.Module):
        def __init__(self, input_size=27, hidden_size=8):
            super().__init__()
            self.active = nn.Tanh()
            
            # Memory updater
            self.memory_stream = nn.Sequential(
                nn.Linear(input_size + hidden_size, 32),
                self.active,
                nn.Linear(32, 16),
                self.active,
                nn.Linear(16, hidden_size),
                self.active
            )
            
            # Logit extractor
            self.extractor = nn.Sequential(
                    nn.Linear(hidden_size, 16),
                    nn.Tanh(),              # activation BETWEEN hidden layers ✓
                    nn.Linear(16, 32),
                    nn.Tanh(),
                    nn.Linear(32, input_size)
                    # NO activation after the final layer ✗
            )
            
        def forward(self, x, h):
            combined = torch.cat((x, h), dim=1)
            new_h = self.memory_stream(combined)
            logits = self.extractor(new_h)
            return logits, new_h

# --- 3. The Sequence Manager ---
class test_model(nn.Module):
    def __init__(self, input_size=27, hidden_size=8, eos_index=26):
        super().__init__()
        self.encoder = RNNencoder(input_size, hidden_size)
        self.decoder = RNNdecoder(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.eos_index = eos_index
        
    def forward(self, seq_inputs, max_length=10, stop_on_eos=True):
        batch_size = seq_inputs.size(0)
        seq_length = seq_inputs.size(1)
        device = seq_inputs.device
        dtype = seq_inputs.dtype
        
        # --- ENCODER PHASE ---
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        for t in range(seq_length):
            x_t = seq_inputs[:, t, :]
            h = self.encoder(x_t, h)
            
        context_vector = h 
        
        # --- DECODER PHASE ---
        h = context_vector
        curr_input = torch.zeros(batch_size, seq_inputs.size(2), device=device, dtype=dtype)
        outputs = torch.empty(batch_size, max_length, seq_inputs.size(2), device=device, dtype=dtype)
        
        for t in range(max_length):
            logits, h = self.decoder(curr_input, h)
            outputs[:, t, :] = logits
            
            # Stop logic
            predicted_index = torch.argmax(logits, dim=-1)
            if stop_on_eos and (predicted_index == self.eos_index).all():
                outputs = outputs[:, : t + 1, :]
                break
                
            # Autoregressive step
            curr_input = torch.softmax(logits, dim=-1)
            
        return outputs

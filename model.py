import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. The Encoder ---
class RNNencoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=8):
        super().__init__()
        self.active = nn.Tanh()
        self.model = nn.Sequential(
            nn.Linear(input_size + hidden_size, 12),
            self.active,
            nn.Linear(12, 10),
            self.active,
            nn.Linear(10, hidden_size),
            self.active # Bounds final memory
        )
        
    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        return self.model(combined)

# --- 2. The Decoder ---
class RNNdecoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=8):
        super().__init__()
        self.active = nn.Tanh()
        
        # Memory updater
        self.memory_stream = nn.Sequential(
            nn.Linear(input_size + hidden_size, 12),
            self.active,
            nn.Linear(12, 10),
            self.active,
            nn.Linear(10, hidden_size),
            self.active
        )
        
        # Logit extractor
        self.extractor = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        new_h = self.memory_stream(combined)
        logits = self.extractor(new_h)
        return logits, new_h

# --- 3. The Sequence Manager ---
class CustomSeq2Seq(nn.Module):
    def __init__(self, input_size=6, hidden_size=8, eos_index=5):
        super().__init__()
        self.encoder = RNNencoder(input_size, hidden_size)
        self.decoder = RNNdecoder(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.eos_index = eos_index
        
    def forward(self, seq_inputs, max_length=10):
        batch_size = seq_inputs.size(0)
        seq_length = seq_inputs.size(1)
        
        # --- ENCODER PHASE ---
        h = torch.zeros(batch_size, self.hidden_size)
        for t in range(seq_length):
            x_t = seq_inputs[:, t, :]
            h = self.encoder(x_t, h)
            
        context_vector = h 
        
        # --- DECODER PHASE ---
        h = context_vector 
        curr_input = torch.zeros(batch_size, seq_inputs.size(2)) 
        outputs = []
        
        for t in range(max_length):
            logits, h = self.decoder(curr_input, h)
            outputs.append(logits)
            
            # Stop logic
            predicted_index = torch.argmax(logits, dim=-1)
            if (predicted_index == self.eos_index).all():
                break
                
            # Autoregressive step
            curr_input = torch.softmax(logits, dim=-1)
            
        return torch.stack(outputs, dim=1)

# --- 4. The Training Execution ---
if __name__ == "__main__":
    # Settings
    INPUT_DIM = 6
    HIDDEN_DIM = 8
    EOS_IDX = 5
    
    # Initialize
    model = CustomSeq2Seq(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, eos_index=EOS_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    
    # Dummy Training Data (Batch of 1, 3 Steps, 6 Dimensions)
    # Target: [Data, Data, EOS]
    dummy_input = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # EOS Token
    ]])
    
    # Target Class Indices for the Loss Function: [0, 1, 5]
    targets = torch.tensor([[0, 1, 5]]) 
    
    # --- TRAINING LOOP (1 Iteration) ---
    optimizer.zero_grad()                         # Clear old gradients
    
    predictions = model(dummy_input, max_length=3) # Forward Pass
    
    # Reshape for PyTorch CrossEntropyLoss (Requires [Batch * Seq_Len, Classes])
    flat_predictions = predictions.view(-1, INPUT_DIM)
    flat_targets = targets.view(-1)
    
    loss = loss_function(flat_predictions, flat_targets) # Calculate Error
    loss.backward()                                      # Backpropagation Through Time
    optimizer.step()                                     # Update Weights
    
    print("Training Step Complete.")
    print(f"Loss: {loss.item():.4f}")
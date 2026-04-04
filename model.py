import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RNNencoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=64):
        super().__init__()
        self.x_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.h_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, x, h):
        x_feat = self.x_net(x)
        h_feat = self.h_net(h)
        combined = x_feat + h_feat
        return torch.tanh(combined + h)
    

class RNNdecoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=64):
        super().__init__()
        self.memory_stream = nn.Sequential(
            nn.Linear(input_size + hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, hidden_size),
            nn.Tanh()
        )
        
        self.extractor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, input_size)
        )
        
    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        raw_new_h = self.memory_stream(combined)
        new_h = torch.tanh(raw_new_h + h) 
        
        logits = self.extractor(new_h)
        return logits, new_h

class test_model(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, eos_index=26):
        super().__init__()
        self.encoder = RNNencoder(input_size, hidden_size)
        self.decoder = RNNdecoder(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.eos_index = eos_index
        
    def forward(self, seq_inputs, max_length=10, stop_on_eos=True, teacher_forcing_ratio=.5):
        batch_size = seq_inputs.size(0)
        device = seq_inputs.device
        dtype = seq_inputs.dtype

        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        for t in range(seq_inputs.size(1)):
            h = self.encoder(seq_inputs[:, t, :], h)


        curr_input = torch.zeros(batch_size, self.input_size, device=device, dtype=dtype)
        outputs = []

        for t in range(max_length):
            logits, h = self.decoder(curr_input, h)
            outputs.append(logits.unsqueeze(1))

            use_teacher = (
                self.training
                and teacher_forcing_ratio > 0.0
                and t < seq_inputs.size(1)
                and random.random() < teacher_forcing_ratio
            )

            if use_teacher:
                curr_input = seq_inputs[:, t, :]
            else:
                curr_input = torch.softmax(logits, dim=-1)

            if stop_on_eos and not self.training:
                if (torch.argmax(logits, dim=-1) == self.eos_index).all():
                    break

        return torch.cat(outputs, dim=1)
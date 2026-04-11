import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RNNencoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=64):
        super().__init__()
        self.tanh = nn.Tanh()
        self.x_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
        )
        self.h_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Sigmoid(),
        )
        
    def forward(self, x, h, up, ):
        x_feat = self.x_net(x)
        h_feat = self.h_net(h)
        candidate = self.candidate(torch.cat([x_feat, h_feat, h], dim=1))
        update = self.update_gate(torch.cat([x_feat, h_feat, up], dim=1))
        new_h = (1.0 - update) * h + update * candidate
        return new_h, update
    

class RNNdecoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=64):
        super().__init__()
        self.tanh = nn.Tanh()
        self.candidate = nn.Sequential(
            nn.Linear(input_size + hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, hidden_size),
            nn.Tanh(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        self.extractor = nn.Sequential(
            nn.Linear(hidden_size,512),
            nn.GELU(),
            nn.Linear(512, input_size)
        )
    
    def forward(self, x, h, up):
        combined = torch.cat((x, h), dim=1)
        candidate = self.candidate(combined)
        update = self.update_gate(torch.cat((combined, up), dim=1))
        new_h = (1.0 - update) * h + update * candidate
        logits = self.extractor(new_h)
        return logits, new_h, update

class test_model(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, eos_index=26):
        super().__init__()
        self.encoder = RNNencoder(input_size, hidden_size)
        self.decoder = RNNdecoder(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.eos_index = eos_index
        
        latent_dim = int(hidden_size*0.75)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)
        self.z_to_hidden = nn.Linear(latent_dim, hidden_size)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, seq_inputs, max_length=30, stop_on_eos=True, teacher_forcing_ratio=.5):
        batch_size = seq_inputs.size(0)
        device = seq_inputs.device
        dtype = seq_inputs.dtype

        h  = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        up = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        stack = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        for t in range(seq_inputs.size(1)):
            h,up = self.encoder(seq_inputs[:, t, :], h,up,)

        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        h = self.z_to_hidden(z)
        curr_input = torch.zeros(batch_size, self.input_size, device=device, dtype=dtype)
        outputs = []

        for t in range(max_length):
            logits, h, up = self.decoder(curr_input, h, up,)
            outputs.append(logits.unsqueeze(1))
            pred_idx = torch.argmax(logits, dim=-1)

            use_teacher = (
                self.training
                and teacher_forcing_ratio > 0.0
                and t < seq_inputs.size(1)
                and random.random() < teacher_forcing_ratio
            )

            if use_teacher:
                curr_input = seq_inputs[:, t, :]
            else:
                if self.training:
                    curr_input = F.gumbel_softmax(logits, tau=1.0, hard=True)
                else:
                    curr_input = F.one_hot(pred_idx, num_classes=self.input_size).to(dtype=dtype)

            if stop_on_eos and not self.training:
                if (pred_idx == self.eos_index).all():
                    break

        return torch.cat(outputs, dim=1),mu, logvar
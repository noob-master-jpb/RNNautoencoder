import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RNNencoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, dropout=0.1):
        super().__init__()
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)
        self.candidate = nn.Linear(hidden_size * 3, hidden_size)
        self.update_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.reset_gate = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, x, h, up):
        x_feat = F.gelu(self.x_proj(x))
        h_feat = F.gelu(self.h_proj(h))

        gate_in = torch.cat([x_feat, h_feat, up], dim=1)
        reset = torch.sigmoid(self.reset_gate(gate_in))
        candidate = torch.tanh(self.candidate(torch.cat([x_feat, h_feat, reset * h], dim=1)))
        update = torch.sigmoid(self.update_gate(gate_in))

        new_h = (1.0 - update) * h + update * candidate
        return new_h, update
    

class RNNdecoder(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, dropout=0.1):
        super().__init__()
        self.cand_fc1 = nn.Linear(input_size + hidden_size, hidden_size * 4)
        self.cand_fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)

        self.ext_fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.ext_fc2 = nn.Linear(hidden_size * 4, input_size)

    def forward(self, x, h, up):
        combined = torch.cat((x, h), dim=1)
        gate_in = torch.cat((combined, up), dim=1)
        reset = torch.sigmoid(self.reset_gate(gate_in))

        cand_in = torch.cat((x, reset * h), dim=1)
        cand_hidden = F.gelu(self.cand_fc1(cand_in))
        candidate = torch.tanh(self.cand_fc2(cand_hidden))

        update = torch.sigmoid(self.update_gate(gate_in))
        new_h = (1.0 - update) * h + update * candidate

        logits_hidden = F.gelu(self.ext_fc1(new_h))
        logits = self.ext_fc2(logits_hidden)
        return logits, new_h, update

class test_model(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, eos_index=26, dropout=0.1, input_dropout=0.05):
        super().__init__()
        self.encoder = RNNencoder(input_size, hidden_size, dropout=dropout)
        self.decoder = RNNdecoder(input_size, hidden_size, dropout=dropout)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.eos_index = eos_index
        self.start_token = nn.Parameter(torch.zeros(1, input_size))
        self.max_stack_len = 30
        self.attn_q = nn.Linear(hidden_size, hidden_size)
        self.attn_k = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, hidden_size)
        
        latent_dim = int(hidden_size*0.75)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)
        self.z_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.Tanh(),
        )
        self.latent_skip = nn.Linear(hidden_size, hidden_size)
        self.latent_gate = nn.Parameter(torch.tensor(0.0))
    
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, seq_inputs, max_length=30, stop_on_eos=True, teacher_forcing_ratio=.5):
        batch_size = seq_inputs.size(0)
        device = seq_inputs.device
        dtype = seq_inputs.dtype

        h  = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        up = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        h_stack = []
        for t in range(seq_inputs.size(1)):
            h, up = self.encoder(seq_inputs[:, t, :], h, up)
            if len(h_stack) < self.max_stack_len:
                h_stack.append(h)
            else:
                h_stack.pop(0)
                h_stack.append(h)

        h_enc = h
        h_final = torch.tanh(h_enc)
        if len(h_stack) == 0:
            h_stack.append(h_final)

        h_stack_tensor = torch.stack(h_stack, dim=1)
        h_stack_plus = torch.cat([h_stack_tensor, h_final.unsqueeze(1)], dim=1)

        q = self.attn_q(h_final).unsqueeze(1)
        k = self.attn_k(h_stack_plus)
        v = self.attn_v(h_stack_plus)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_context = torch.matmul(attn_weights, v).squeeze(1)
        attn_context = torch.tanh(attn_context)
        h_enc = attn_context + h_final
        mu = self.mu(h_enc)
        logvar = self.logvar(h_enc)
        z = self.reparameterize(mu, logvar)
        h = self.z_to_hidden(z)
        h = h + torch.sigmoid(self.latent_gate) * self.latent_skip(h_enc)
        curr_input = self.start_token.expand(batch_size, -1).to(device=device, dtype=dtype)
        outputs = []

        for t in range(max_length):
            logits, h, up = self.decoder(curr_input, h, up)
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
                    # Use softmax for better gradient flow during training
                    probs = torch.softmax(logits, dim=-1)
                    curr_input = probs @ torch.eye(self.input_size, device=device, dtype=dtype)
                else:
                    curr_input = F.one_hot(pred_idx, num_classes=self.input_size).to(dtype=dtype)

            if stop_on_eos and not self.training:
                if (pred_idx == self.eos_index).all():
                    break

        return torch.cat(outputs, dim=1),mu, logvar
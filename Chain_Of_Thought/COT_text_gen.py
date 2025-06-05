import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer

class ChainOfThought(nn.Module):
    def __init__(self, vocab_size, vocab, idx_to_char, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(ChainOfThought, self).__init__()
        self.transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.idx_to_char = idx_to_char
        self.d_model = d_model

        
    
    def generate(self, question, max_length=50, temperature=1.0):
        cot_prmpt = question + " Let's think step by step."
        input_idx=[]
        for char in cot_prmpt:
            if char in self.vocab:
                input_idx.append(self.vocab[char])
            else:
                input_idx.append(self.vocab['<unk>']) 
        input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0)
        generated_text = cot_prmpt
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                seq_len = input_tensor.size(1)
                src_mask = None  
                tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).unsqueeze(0)
                
                output = self.transformer(input_tensor, input_tensor, src_mask, tgt_mask)
                next_token_logits = output[:, -1, :]
                
                if temperature != 1.0 and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                top_k_indices = torch.topk(next_token_logits, k=min(5, self.vocab_size)).indices
                next_token_idx = top_k_indices[0, 0].item()
                
                generated_text += self.idx_to_char[next_token_idx]
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_idx]], dtype=torch.long)], dim=1) 
                return generated_text       
    def forward(self, question, max_length=150, temperature=1.0):
        return self.generate(question, max_length, temperature)
    def encoder(self, text: str):
        input_idx = [self.vocab.get(char, self.vocab['<unk>']) for char in text]
        return torch.tensor(input_idx, dtype=torch.long).unsqueeze(0)
    def decoder(self, input_tensor: torch.Tensor, max_length: int = 50, temperature: float = 1.0):
        generated_text = ""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                seq_len = input_tensor.size(1)
                src_mask = None  
                tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).unsqueeze(0)
                
                output = self.transformer(input_tensor, input_tensor, src_mask, tgt_mask)
                next_token_logits = output[:, -1, :]
                
                if temperature != 1.0 and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                top_k_probs, top_k_indices = torch.topk(next_token_logits, k=min(5, self.vocab_size))
                next_token_idx = top_k_indices[0, 0].item()
                
                generated_text += self.idx_to_char[next_token_idx]
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_idx]], dtype=torch.long)], dim=1) 
        return generated_text




        
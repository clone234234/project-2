import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer

class ChainOfThought(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(ChainOfThought, self).__init__()
        self.transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.vocab_size = vocab_size
        self.d_model = d_model
    

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.transformer(src, tgt, src_mask, tgt_mask)
    def generate_with_reasoning(self, src, max_length=50, temperature=1.0):
        self.eval()
        src = src.unsqueeze(0)
    

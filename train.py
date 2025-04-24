import torch
import torch.nn as nn 
from torch.nn import functional as F
from transformer import Transformer

def preprocess_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]  
        lines = [line.split('\t') for line in lines]
        chars = sorted(list(set(''.join([line[0] for line in lines] + [line[1] for line in lines]))))
        vocab = {char: i for i, char in enumerate(chars)}
        vocab_size = len(vocab)
    return lines, vocab, vocab_size

def generate_mask(src, tgt, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

def batch(batch_data, vocab, device):
    src = [line[0] for line in batch_data]
    tgt = [line[1] for line in batch_data]
    src_tensor = torch.tensor([[vocab[char] for char in seq] for seq in src], dtype=torch.long, device=device)
    tgt_tensor = torch.tensor([[vocab[char] for char in seq] for seq in tgt], dtype=torch.long, device=device)
    return src_tensor, tgt_tensor

def train_transformer(model, data, vocab_size, num_epochs=10, batch_size=32, learning_rate=0.001, vocab=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 




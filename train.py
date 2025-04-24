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



def batch(batch_data, vocab, device):
    src = [line[0] for line in batch_data]
    tgt = [line[1] for line in batch_data]
    src_tensor = torch.tensor([[vocab[char] for char in seq] for seq in src], dtype=torch.long, device=device)
    tgt_tensor = torch.tensor([[vocab[char] for char in seq] for seq in tgt], dtype=torch.long, device=device)
    return src_tensor, tgt_tensor

def train_transformer(model, data, vocab, num_epochs=10, batch_size=32, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range (0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            src_tensor, tgt_tensor = batch(batch_data, vocab, device)
            
            src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)  
            tgt_mask = (tgt_tensor != 0).unsqueeze(1).unsqueeze(2)  
            
            optimizer.zero_grad()
            output = model(src_tensor, tgt_tensor[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt_tensor[:, 1:].view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data)}')



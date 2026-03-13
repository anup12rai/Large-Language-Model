import torch

def encode_texts(tokenizer, texts, max_len=32):
    encodings = [tokenizer.encode(t).ids for t in texts]
    padded = [e + [0]*(max_len - len(e)) if len(e)<max_len else e[:max_len] for e in encodings]
    return torch.tensor(padded)
import torch

def save_model(model, path="model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model, path="model.pt", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
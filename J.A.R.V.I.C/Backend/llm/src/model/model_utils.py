import torch

def save_model(model, path="model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model, path="model.pt", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def predict(model, tokenizer, text, device="cpu"):
    model.to(device)
    inputs = torch.tensor(tokenizer.encode(text).ids).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.argmax(dim=1).item()
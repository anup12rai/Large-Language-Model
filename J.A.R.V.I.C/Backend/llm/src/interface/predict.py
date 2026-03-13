import torch
from src.preprocessing.clean_text import clean_text
from src.preprocessing.tokenizer import load_tokenizer
from src.preprocessing.encode_data import encode_texts
from src.model.transformer import SimpleTransformer
from src.model.classifier import QueryClassifier
from src.model.model_utils import load_model
from src.utils.constants import LABELS
from src.training.config import config  # optional if using config.yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class QueryPredictor:
    def __init__(self, model_path, tokenizer_path, device=DEVICE):
        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Initialize model
        self.transformer = SimpleTransformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            embed_dim=config.get("embed_dim", 64),
            num_heads=config.get("num_heads", 4),
            ff_hidden=config.get("ff_hidden", 128),
            num_layers=config.get("num_layers", 2),
            max_len=config.get("max_seq_len", 32)
        )
        self.model = QueryClassifier(self.transformer)
        self.model = load_model(self.model, model_path, device=device)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, query):
        query_clean = clean_text(query)
        encoded = encode_texts(self.tokenizer, [query_clean], max_len=config.get("max_seq_len", 32))
        encoded = encoded.to(self.device)
        with torch.no_grad():
            logits = self.model(encoded)
            pred_idx = torch.argmax(logits, dim=-1).item()
        return LABELS[pred_idx]


# Example usage:
if __name__ == "__main__":
    predictor = QueryPredictor(model_path="../model.pt", tokenizer_path="../data/processed/tokenizer.json")
    
    while True:
        query = input("Enter query (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        label = predictor.predict(query)
        print(f"Predicted Type: {label}\n")
from src.preprocessing.clean_text import clean_text
from src.preprocessing.tokenizer import load_tokenizer
from src.preprocessing.encode_data import encode_texts
from src.model.transformer import SimpleTransformer
from src.model.classifier import QueryClassifier
from src.model.model_utils import load_model
import torch
from src.utils.constants import LABELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = load_tokenizer()
transformer = SimpleTransformer(vocab_size=tokenizer.get_vocab_size())
model = QueryClassifier(transformer)
model = load_model(model, "model.pt", device=DEVICE)
model.to(DEVICE)

def classify_query(query):
    query = clean_text(query)
    encoded = encode_texts(tokenizer, [query])
    encoded = encoded.to(DEVICE)
    with torch.no_grad():
        logits = model(encoded)
        pred = torch.argmax(logits, dim=-1).item()
    return LABELS[pred]
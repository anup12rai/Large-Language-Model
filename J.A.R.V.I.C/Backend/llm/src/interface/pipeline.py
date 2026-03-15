import torch
from src.preprocessing.clean_text import clean_text
from src.preprocessing.tokenizer import load_tokenizer
from src.preprocessing.encode_data import encode_texts
from src.model.transformer import SimpleTransformer
from src.model.classifier import QueryClassifier
from src.model.model_utils import load_model
from src.utils.constants import LABELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = load_tokenizer("data/processed/tokenizer.json")

transformer = SimpleTransformer(vocab_size=tokenizer.get_vocab_size())
model = QueryClassifier(transformer)

model = load_model(model, "model.pt", DEVICE)
model.to(DEVICE)
model.eval()


def classify_query(query):

    query = clean_text(query)

    encoded = encode_texts(tokenizer, [query])
    encoded = encoded.to(DEVICE)

    with torch.no_grad():
        logits = model(encoded)
        pred = torch.argmax(logits, dim=1).item()

    return LABELS[pred]

if __name__ == "__main__":
    while True:
        query = input("Enter query (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        label = classify_query(query)
        print(f"Predicted Type: {label}\n")
# Filename: query_classifier_user_input.py

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# =========================
# Step 1: Prepare Data
# =========================
# Example data - you can expand this dataset
data = {
    "query": [
        "Turn on the fan",
        "Start the meeting lights",
        "What's the weather today?",
        "Who is the president of Nepal?",
        "Is the server running?",
        "Check room temperature",
        "Open the door",
        "Tell me a joke"
    ],
    "label": [
        "automation",
        "automation",
        "general",
        "general",
        "realtime",
        "realtime",
        "automation",
        "general"
    ]
}

df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])  # Converts text labels to numbers

# =========================
# Step 2: Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def encode(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

# =========================
# Step 3: Dataset
# =========================
class QueryDataset(Dataset):
    def __init__(self, df):
        self.texts = df['query'].tolist()
        self.labels = df['label_enc'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = encode([self.texts[idx]])
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0), torch.tensor(self.labels[idx])

dataset = QueryDataset(df)

# Collate function to pad variable-length sequences
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return input_ids, attention_mask, labels

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# =========================
# Step 4: Model
# =========================
class QueryClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.fc(cls_token)

model = QueryClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# =========================
# Step 5: Training
# =========================
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in loader:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# Save the model and label encoder
torch.save(model.state_dict(), "query_classifier_model.pth")
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# =========================
# Step 6: User Input Loop
# =========================
print("\n✅ Model trained! You can now enter queries to classify them.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter your query: ")
    if user_input.lower() == "exit":
        break

    # Tokenize and pad
    enc = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    
    model.eval()
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
        pred_idx = torch.argmax(logits, dim=1).item()
        category = le.inverse_transform([pred_idx])[0]
    
    print(f"Category: {category}\n")
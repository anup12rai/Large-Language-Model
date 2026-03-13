import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data/raw"

def load_data():
    texts, labels = [], []
    for label, file_name in enumerate(["general.txt", "realtime.txt", "automation.txt"]):
        path = os.path.join(DATA_DIR, file_name)
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            texts.extend(lines)
            labels.extend([label]*len(lines))
    return texts, labels

def get_train_val_split(test_size=0.1, random_state=42):
    texts, labels = load_data()
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)
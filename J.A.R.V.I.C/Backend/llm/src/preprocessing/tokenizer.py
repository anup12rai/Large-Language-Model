from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_tokenizer(texts, vocab_size=2000, save_path="data/processed/tokenizer.json"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer

def load_tokenizer(path="data/processed/tokenizer.json"):
    return Tokenizer.from_file(path)
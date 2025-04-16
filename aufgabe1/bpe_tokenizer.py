from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer 
from tokenizers.pre_tokenizers import Whitespace

# Trainiert einen BPE Tokenizer und speichert ihn als Datei
def train_tokenizer(corpus_path, vocab_size, output_path):
    
    # Initialisierung
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(output_path)
    print("Tokenizer mit", tokenizer.get_vocab_size(), "Tokens gespeichert: ", output_path)
    
    return tokenizer

def tokenize_text(tokenizer_path, text):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    encoding = tokenizer.encode(text)
    return encoding.tokens
    
    
if __name__ == "__main__":
    train_tokenizer("./datasets/bible_de.txt", 1000, "./trained_tokenizers/bible_de_bpe.json")
    
    tokens = tokenize_text("./trained_tokenizers/bible_de_bpe.json", "Das ist ein Test.")
    print("Anzahl Tokens: ", len(tokens))
    print("Tokens: ", tokens)
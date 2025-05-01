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

def tokenize_output(tokenizer_path, text):
    tokens = tokenize_text(tokenizer_path, text)
    print("--------------------------------------------------------------------------")
    print("Tokenizer: ", tokenizer_path)
    print("Anzahl Tokens:", len(tokens))
    print("Tokens:", tokens)
    
    
    
if __name__ == "__main__":
    # Trainieren der Tokenizer
    train_tokenizer("./datasets/bible_de.txt", 1000, "./trained_tokenizers/bible_de_bpe.json")
    train_tokenizer("./datasets/bible_de.txt", 3000, "./trained_tokenizers/bible_de_bpe_3k.json")
    
    train_tokenizer("./datasets/bible_en.txt", 1000, "./trained_tokenizers/bible_en_bpe.json")
    train_tokenizer("./datasets/bible_en.txt", 3000, "./trained_tokenizers/bible_en_bpe_3k.json")
    
    train_tokenizer("./datasets/bible_de-en.txt", 1000, "./trained_tokenizers/bible_de-en_bpe.json")
    train_tokenizer("./datasets/bible_de-en.txt", 3000, "./trained_tokenizers/bible_de-en_bpe_3k.json")
    
    # Testen der Tokenizer
    tokenize_output("./trained_tokenizers/bible_de_bpe.json", "Das ist ein Test.")
    tokenize_output("./trained_tokenizers/bible_de_bpe_3k.json", "Das ist ein Test.")
    
    tokenize_output("./trained_tokenizers/bible_en_bpe.json", "This is a test.")
    tokenize_output("./trained_tokenizers/bible_en_bpe_3k.json", "This is a test.")
    
    tokenize_output("./trained_tokenizers/bible_de-en_bpe.json", "Das ist ein Test.")
    tokenize_output("./trained_tokenizers/bible_de-en_bpe_3k.json", "Das ist ein Test.")
    tokenize_output("./trained_tokenizers/bible_de-en_bpe.json", "This is a test.")
    tokenize_output("./trained_tokenizers/bible_de-en_bpe_3k.json", "This is a test.")
    tokenize_output("./trained_tokenizers/bible_de-en_bpe.json", "This is ein Test.")
    tokenize_output("./trained_tokenizers/bible_de-en_bpe_3k.json", "This is ein Test.")
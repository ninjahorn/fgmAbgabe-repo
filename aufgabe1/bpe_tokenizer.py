import json

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        # Fuer unbekannte Woerter
        self.special_tokens = ["[UNK]"]
    
    def train(self, corpus_path):
        print("Training BPE tokenizer on corpus:", corpus_path, "with vocab size:", self.vocab_size)
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        words = text.split()
        
        split_words = []
        for word in words:
            chars = list(word) + ['</w>']
            split_words.append(chars)
            
        all_characters = set()
        for word in split_words:
            all_characters.update(word)
            
        for char in all_characters:
            self.vocab[char] = len(self.vocab)
            
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                
        num_merges = min(self.vocab_size - len(self.vocab), 10000)
        for i in range(num_merges):
            if i % 100 == 0:
                print("Merge iteration: ", i/num_merges)
            pairs = self._count_token_pairs(split_words)
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token
            split_words = self._apply_merge(split_words, best_pair, new_token)
            
        print("Finished training with ", len(self.vocab), " tokens in vocabulary.")
    
    def save(self, output_path):
        serializable_merges = {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()}
        
        data = {
            'vocab': self.vocab,
            'merges': serializable_merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print("Tokenizer with ", len(self.vocab), "tokens saved to:", output_path)
    
    def tokenize(self, text):
        words = text.split()
        result = []
        
        for word in words:
            word_tokens = list(word) + ['</w>']
            
            while True:
                possible_merges = []
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    if pair in self.merges:
                        possible_merges.append((i, pair))
                
                if not possible_merges:
                    break
                
                i, pair = possible_merges[0]
                new_token = self.merges[pair]
                word_tokens = word_tokens[:i] + [new_token] + word_tokens[i+2:]
                
            # Entfernen von </w> und leeren Tokens fuer saubere Ausgabe
            cleaned_tokens = [token.replace('</w>', '') for token in word_tokens]
            cleaned_tokens = [token for token in cleaned_tokens if token]
            result.extend(cleaned_tokens)
            
        return result
    
    
    # Laedt Tokenizer aus einer json Datei
    @classmethod
    def load(cls, tokenizer_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data.get('vocab_size', 1000))
        tokenizer.vocab = {k: int(v) if isinstance(v, str) else v for k, v in data['vocab'].items()}
        tokenizer.merges = {tuple(k.split('|')): v for k, v in data['merges'].items()}
        tokenizer.special_tokens = data.get('special_tokens', ["[UNK]"])
        return tokenizer
    
    # Die Haeufigkeitder von Paaren in den Woertern wird gezaehlt
    def _count_token_pairs(self, split_words):
        pairs = {}
        for word in split_words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                
                if pair in pairs:
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1
        return pairs
    
    # Paare werden mit neuem Token zusammengefasst
    def _apply_merge(self, split_words, pair, new_token):
        result = []
        first, second = pair
        
        for word in split_words:
            new_word = []
            i = 0
            
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            result.append(new_word)
        return result
    
# Funktion fuer Ausgabe der Tokenisierung fuer einen Text und bestimmten Tokenizer
def tokenize_and_print(text, tokenizer, tokenizer_name="Tokenizer"):
    tokens = tokenizer.tokenize(text)
    print("--------------------------------------------------")
    print("Tokenizer:", tokenizer_name)
    print("Tokens:", tokens)
    print("Anzahl Tokens:", len(tokens))
    

if __name__ == "__main__":
    # tokenizer_bible_de = BPETokenizer(vocab_size=1000)
    # tokenizer_bible_de.train("./datasets/bible_de.txt")
    # tokenizer_bible_de.save("./trained_tokenizers/bible_de_bpe.json")
    
    # tokenizer_bible_en = BPETokenizer(vocab_size=1000)
    # tokenizer_bible_en.train("./datasets/bible_en.txt")
    # tokenizer_bible_en.save("./trained_tokenizers/bible_en_bpe.json")
    
    # tokenizer_bible_ende = BPETokenizer(vocab_size=1000)
    # tokenizer_bible_ende.train("./datasets/bible_de-en.txt")
    # tokenizer_bible_ende.save("./trained_tokenizers/bible_de-en_bpe.json")
    
    # tokenizer_de = BPETokenizer(vocab_size=1000)
    # tokenizer_de.train("./datasets/amt_de.txt")
    # tokenizer_de.save("./trained_tokenizers/amt_de_bpe.json")
    
    # tokenizer_en = BPETokenizer(vocab_size=1000)
    # tokenizer_en.train("./datasets/amt_en.txt")
    # tokenizer_en.save("./trained_tokenizers/amt_en_bpe.json")
    
    tokenizer_bible_de = BPETokenizer.load("./trained_tokenizers/bible_de_bpe.json")
    tokenize_and_print("Das ist ein Test", tokenizer_bible_de, "German bible tokenizer")
    tokenize_and_print("Die Katze sitz auf dem Dach", tokenizer_bible_de, "German bible tokenizer")
    tokenize_and_print("The cat sits on the roof", tokenizer_bible_de, "German bible tokenizer")
    tokenize_and_print("Jesus ist am Kreuz gestorben", tokenizer_bible_de, "German bible tokenizer")
    tokenize_and_print("Jesus died on the cross", tokenizer_bible_de, "German bible tokenizer")
    
    tokenizer_bible_en = BPETokenizer.load("./trained_tokenizers/bible_en_bpe.json")
    tokenize_and_print("Das ist ein Test", tokenizer_bible_en, "English bible tokenizer")
    tokenize_and_print("Die Katze sitz auf dem Dach", tokenizer_bible_en, "English bible tokenizer")
    tokenize_and_print("The cat sits on the roof", tokenizer_bible_en, "English bible tokenizer")
    tokenize_and_print("Jesus ist am Kreuz gestorben", tokenizer_bible_en, "English bible tokenizer")
    tokenize_and_print("Jesus died on the cross", tokenizer_bible_en, "English bible tokenizer")
    
    tokenizer_bible_ende = BPETokenizer.load("./trained_tokenizers/bible_de-en_bpe.json")
    tokenize_and_print("Das ist ein Test", tokenizer_bible_ende, "German-English bible tokenizer")
    tokenize_and_print("Die Katze sitz auf dem Dach", tokenizer_bible_ende, "German-English bible tokenizer")
    tokenize_and_print("The cat sits on the roof", tokenizer_bible_ende, "German-English bible tokenizer")
    tokenize_and_print("Jesus ist am Kreuz gestorben", tokenizer_bible_ende, "German-English bible tokenizer")
    tokenize_and_print("Jesus died on the cross", tokenizer_bible_ende, "German-English bible tokenizer")
    
    tokenizer_amt_de = BPETokenizer.load("./trained_tokenizers/amt_de_bpe.json")
    tokenize_and_print("Das ist ein Test", tokenizer_amt_de, "German amt tokenizer")
    tokenize_and_print("Die Katze sitz auf dem Dach", tokenizer_amt_de, "German amt tokenizer")
    tokenize_and_print("The cat sits on the roof", tokenizer_amt_de, "German amt tokenizer")
    tokenize_and_print("Jesus ist am Kreuz gestorben", tokenizer_amt_de, "German amt tokenizer")
    tokenize_and_print("Jesus died on the cross", tokenizer_amt_de, "German amt tokenizer")
    
    tokenizer_amt_en = BPETokenizer.load("./trained_tokenizers/amt_en_bpe.json")
    tokenize_and_print("Das ist ein Test", tokenizer_amt_en, "English amt tokenizer")
    tokenize_and_print("Die Katze sitz auf dem Dach", tokenizer_amt_en, "English amt tokenizer")
    tokenize_and_print("The cat sits on the roof", tokenizer_amt_en, "English amt tokenizer")
    tokenize_and_print("Jesus ist am Kreuz gestorben", tokenizer_amt_en, "English amt tokenizer")
    tokenize_and_print("Jesus died on the cross", tokenizer_amt_en, "English amt tokenizer")
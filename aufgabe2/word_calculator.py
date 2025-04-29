import numpy as np
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch

def load_glove_model(file_path):
    print("Loading GloVe model...")
    model = KeyedVectors.load_word2vec_format(file_path, binary=False, no_header=True)
    
    print("GloVe model loaded. Vocabulary size:", len(model.key_to_index))
    return model

def calculate_with_glove(expression, model):
    tokens = expression.split()
    
    result_vector = None
    current_op = "+"
    
    for token in tokens:
        if token == "+" or token == "-":
            current_op = token
        elif token.lower() in model.key_to_index:
            word_vector = model[token.lower()]
            if result_vector is None:
                result_vector = word_vector
            else:
                if current_op == "+":
                    result_vector = result_vector + word_vector
                elif current_op == "-":
                    result_vector = result_vector - word_vector
        else:
            print("Warning: Word not found in model vocabulary: ", token)
    
    if result_vector is not None:
        similar_words = model.similar_by_vector(result_vector, topn=5)
        return similar_words
    else:
        return []
    

def load_transformer_model():
    print("Loading Transformer model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("Transformer model loaded.")
    return model, tokenizer

def get_bert_embedding(word, model, tokenizer):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    token_count = len(tokenizer.tokenize(word))
    vecs = outputs.last_hidden_state[0, 1:1+token_count]
    return vecs.mean(dim=0).cpu().numpy()

def calculate_with_transformer(expression, model, tokenizer, vocab):
    tokens = expression.split()
    result = None
    current_op = "+"

    for token in tokens:
        if token == "+" or token == "-":
            current_op = token
        else:
            word_vector = get_bert_embedding(token.lower(), model, tokenizer)
            if result is None:
                result = word_vector
            else:
                if current_op == "+":
                    result = result + word_vector
                elif current_op == "-":
                    result = result - word_vector
    if result is None:
        return []
    
    # Limit für vocab size
    vocab = vocab[:1000]
    similarities = []
    for word in vocab:
        word_vector = get_bert_embedding(word, model, tokenizer)
        similarity = np.dot(result, word_vector) / (np.linalg.norm(result) * np.linalg.norm(word_vector))
        similarities.append((word, float(similarity)))
        
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    
if __name__ == "__main__":
    print("Word Calculator")
    print("---------------")
    
    # Auswahl des Modells
    model_choice = ""
    while model_choice not in ["glove", "transformer"]:
        model_choice = input("Choose embedding model (glove/transformer): ")
        if model_choice == "glove":
            glove_model = load_glove_model("./models_and_vocab/glove.6B.50d.txt")
        elif model_choice == "transformer":
            bert_model, bert_tokenizer = load_transformer_model()
            with open ("./models_and_vocab/google-10000-english.txt") as f:
                vocab = [line.strip() for line in f if line.strip().isalpha() and len(line.strip()) > 2]
        else:
            print("Invalid model choice. Please choose 'glove' or 'transformer'.")
        
    
    while True:
        expression = input("Enter a word expression (E.g. king - man + woman): ")
        
        # For transformer model
        if model_choice == "transformer":
            result = calculate_with_transformer(expression, bert_model, bert_tokenizer, vocab)
            if result:
                print("Results: ")
                for word, similarity in result:
                    print(word, ": ", similarity)
            else:
                print("No results found for this expression.")
        # For GloVe model
        else:
            result = calculate_with_glove(expression, glove_model)
            if result:
                print("Results: ")
                for word, similarity in result:
                    print(word, ": ", similarity)
            else:
                print("No results found for this expression.")

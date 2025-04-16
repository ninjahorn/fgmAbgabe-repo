import numpy as np
from gensim.models import KeyedVectors

def load_glove_model(file_path):
    print("Loading GloVe model...")
    model = KeyedVectors.load_word2vec_format(file_path, binary=False)
    
    print("GloVe model loaded. Vocabulary size:", len(model.key_to_index))
    return model

def calculate_with_glove(expression, model):
    return result


if __name__ == "__main__":
    print("Word Calculator")
    print("---------------")
    
    # Auswahl des Modells
    model_choice = input("Choose embedding model (glove/transformer): ")
    if model_choice == "glove":
        glove_model = load_glove_model("path/to/glove.6B.300d.txt")
    elif model_choice == "transformer":
        print("Loading Transformer model...")
    else:
        print("Invalid model choice. Please choose 'glove' or 'transformer'.")
    
    while True:
        expression = input("Enter a word expression (E.g. king - man + woman): ")
        
        if model_choice == "glove":
            result = calculate_with_glove(expression, glove_model)
            
            if result:
                print("Results: ")
                for word, similarity in result:
                    print(word, ": ", similarity)
            else:
                print("No results found for this expression.")
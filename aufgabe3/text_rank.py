import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

# Text laden und in Saetze aufteilen
def load_and_split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence.split()) > 3]
    return sentences

def create_tfidf_representation(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix

def create_glove_representation(sentences):
    glove_model = KeyedVectors.load_word2vec_format("../aufgabe2/models_and_vocab/glove.6B.50d.txt", binary=False, no_header=True)
    sentence_vectors = []
    for sentence in sentences:
        words = [word.lower() for word in sentence.split()]
        word_vectors = [glove_model[word] for word in words if word in glove_model]
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(glove_model.vector_size)
        sentence_vectors.append(sentence_vector)
    return np.array(sentence_vectors)

def create_bert_representation(sentences):
    print("Loading BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Create BERT embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

def create_similarity_graph_tfidf(tfidf_matrix, threshold=0.2):
    n_sentences = tfidf_matrix.shape[0]
    graph = nx.Graph()
    
    for i in range(n_sentences):
        graph.add_node(i)
        
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            similarity = tfidf_matrix[i].dot(tfidf_matrix[j].T).toarray()[0, 0]
            norm_i = np.sqrt(tfidf_matrix[i].dot(tfidf_matrix[i].T).toarray()[0, 0])
            norm_j = np.sqrt(tfidf_matrix[j].dot(tfidf_matrix[j].T).toarray()[0, 0])
            
            if norm_i > 0 and norm_j > 0:
                similarity = similarity / (norm_i * norm_j)
                
                if similarity > threshold:
                    graph.add_edge(i, j, weight=similarity)
    return graph

def create_similarity_graph_glove(embeddings, threshold=0.2):
    n_sentences = embeddings.shape[0]
    graph = nx.Graph()
    
    for i in range(n_sentences):
        graph.add_node(i)
        
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            similarity = np.dot(embeddings[i], embeddings[j])
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            
            if norm_i > 0 and norm_j > 0:
                similarity = similarity / (norm_i * norm_j)
                
                if similarity > threshold:
                    graph.add_edge(i, j, weight=similarity)
    return graph

def create_similarity_graph_embeddings(embeddings, threshold=0.2):
    n_sentences = embeddings.shape[0]
    graph = nx.Graph()
    
    for i in range(n_sentences):
        graph.add_node(i)
        
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            similarity = np.dot(embeddings[i], embeddings[j])
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            
            if norm_i > 0 and norm_j > 0:
                similarity = similarity / (norm_i * norm_j)
                
                if similarity > threshold:
                    graph.add_edge(i, j, weight=similarity)
    return graph

def apply_text_rank(graph):
    scores = nx.pagerank(graph)
    return scores

def extract_top_sentences(sentences, scores, top_n=5):
    ranked_sentences = [(scores[i], i, sentence) for i, sentence in enumerate(sentences)]
    ranked_sentences.sort(reverse=True)
    
    top_sentences = []
    top_indices = []
    
    for score, index, sentence in ranked_sentences[:top_n]:
        top_sentences.append(sentence)
        top_indices.append(index)
        
    return top_sentences, top_indices

def visualize_graph(graph, top_indices=None, title="TextRank Graph"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    
    nx.draw_networkx_nodes(graph, pos, node_size=100, alpha=0.8)
    
    if top_indices:
        nx.draw_networkx_nodes(graph, pos, nodelist=top_indices, node_color='red', node_size=200)
        
    edge_weights = [graph[u][v]['weight']*2 for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.5)
    
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"./plots/{title.replace(' ', '_')}.png")
    plt.close()
    
    

if __name__ == "__main__":
    sentences = load_and_split_text("nlp_wiki_en.txt")
    print("Number of sentences: ", len(sentences))
    
    print("Creating TF-IDF representation...")
    tfidf_matrix = create_tfidf_representation(sentences)
    tfidf_graph = create_similarity_graph_tfidf(tfidf_matrix)
    tfidf_scores = apply_text_rank(tfidf_graph)
    tfidf_summary, tfidf_top_indices = extract_top_sentences(sentences, tfidf_scores)
    
    print("Creating GloVe representation...")
    glove_embeddings = create_glove_representation(sentences)
    glove_graph = create_similarity_graph_glove(glove_embeddings)
    glove_scores = apply_text_rank(glove_graph)
    glove_summary, glove_top_indices = extract_top_sentences(sentences, glove_scores)
    
    print("Creating BERT representation...")
    bert_embeddings = create_bert_representation(sentences)
    bert_graph = create_similarity_graph_embeddings(bert_embeddings)
    bert_scores = apply_text_rank(bert_graph)
    bert_summary, bert_top_indices = extract_top_sentences(sentences, bert_scores)
    
    visualize_graph(tfidf_graph, tfidf_top_indices, title="TF-IDF TextRank Graph")
    visualize_graph(glove_graph, glove_top_indices, title="GloVe TextRank Graph")
    visualize_graph(bert_graph, bert_top_indices, title="BERT TextRank Graph")
    
    # Ergebnisse ausgeben
    print("TF-IDF Summary:")
    for i, sentence in enumerate(tfidf_summary):
        print(i+1, ". ", sentence)
        
    print("\nGloVe Summary:")
    for i, sentence in enumerate(glove_summary):
        print(i+1, ". ", sentence)
        
    print("\nBERT Summary:")
    for i, sentence in enumerate(bert_summary):
        print(i+1, ". ", sentence)
        
    # Compare overlap between methods
    tfidf_glove_common = set(tfidf_summary).intersection(set(glove_summary))
    tfidf_bert_common = set(tfidf_summary).intersection(set(bert_summary))
    glove_bert_common = set(glove_summary).intersection(set(bert_summary))
    all_common = set(tfidf_summary).intersection(set(glove_summary), set(bert_summary))
    
    print("\nOverlap comparison:")
    print(f"TF-IDF vs GloVe: {len(tfidf_glove_common)} of {len(tfidf_summary)} sentences")
    print(f"TF-IDF vs BERT: {len(tfidf_bert_common)} of {len(tfidf_summary)} sentences")
    print(f"GloVe vs BERT: {len(glove_bert_common)} of {len(glove_summary)} sentences")
    print(f"All methods: {len(all_common)} sentences in common")
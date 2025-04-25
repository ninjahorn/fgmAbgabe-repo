# Bericht - Programmieraufgaben - Seminar zu Generativer KI

Dieser Bericht widmet sich den Programmieraufgaben zum Seminar zu Generativer KI

## Aufgabe 1

Für die Implementierung in a) wurden das package "tokenizers" von Hugging Face genutzt. Dieses beinhaltet einen Implementierten BPE Tokenizer...



## Aufgabe 2

In dem models Ordner ist bereits ein 50d glove Modell enthalten. Um zum Beispiel ein größeres Modell, wie 300d, zu nutzen, muss die Modell Datei in den models Ordner verschoben werden und der Pfad in Zeile ... (```glove_model = load_glove_model("./models/glove.6B.50d.txt")```) muss aktualisiert werden.

Mit GloVe Zugriff auf ganzes vokabular, mit trnasformer liefert automodel nur vektoren, aber es gibt keine eingebaute methode, wie model.similar_by_vector um über das gesamte Vokabular die Ähnlichkeit zu berechnen. Deswegen similarity nur bei GloVe.

Dein Ansatz ist grundsätzlich korrekt:

Du berechnest für jedes Wort einen Embedding-Vektor mit BERT, führst die Rechenoperationen im Vektorraum aus und erhältst einen Ergebnisvektor.
Aber:
Im Gegensatz zu GloVe kannst du bei BERT nicht einfach model.similar_by_vector aufrufen, weil BERT kein eingebautes Wörterbuch mit Vektoren für alle Wörter hat.
Das ist ein grundlegender Unterschied zwischen klassischen Embeddings (wie GloVe/Word2Vec) und Transformer-Modellen:

GloVe/Word2Vec
Haben ein fixes Vokabular und für jedes Wort einen Vektor.
Du kannst für jeden beliebigen Vektor die ähnlichsten Wörter im Vokabular finden (similar_by_vector).
Transformer (BERT etc.)
Erzeugen Embeddings on the fly für beliebige Eingaben.
Kein eingebautes Wörterbuch mit allen Wort-Vektoren.
Um das ähnlichste Wort zu finden, musst du für eine Liste von Kandidaten (z.B. häufige Wörter oder dein eigenes Wörterbuch) die Embeddings berechnen und vergleichen (Cosinus-Ähnlichkeit).

Enter a word expression (E.g. king - man + woman): madrid - spain + germany
Results: 
munich :  0.8965941071510315
stuttgart :  0.82121741771698
berlin :  0.8180941343307495
frankfurt :  0.8079747557640076
hamburg :  0.8076310753822327

## Aufgabe 3

Hier steht was zu 3.

## Quellenverzeichnis

blablabla
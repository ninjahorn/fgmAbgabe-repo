# Bericht - Programmieraufgaben - Seminar zu Generativer KI

Dieser Bericht widmet sich den Programmieraufgaben zum Seminar zu Generativer KI. In den Aufgaben befinden sich alle nötigen Informationen zum Ausführen des Codes und zum Vorgehen wärend der Bearbeitung. Alle Aufgaben wurden in Python implementiert. Ausgeführt wurde alles in einer WSL (Windows Subsystem for Linux) mit python3. Zum Ausführen von einigen Programmen wird eventuell eine Installation von Bibliotheken, wie "transformers" oder "torch" benötigt (```pip install ...```).

Tipp: Mit ```Strg + Shift + V``` lässt sich in Visual Studio Code die Markdown Preview öffnen.

## Aufgabe 1

Der BPE Tokenizer Tokenizer wurde als Klasse in der Datei ```bpe_tokenizer.py``` implementiert. Dazu gehören einige Funktionen, wie ```train()```, welche den Tokenizer auf einem gegebenen Corpus trainiert. Dazu gehören auch die beiden Hilfsfunktionen ```_count_token_pairs()``` und ```_apply_merge()```. Mit ```save()``` lässt sich ein trainierter Tokenizer als json Datei speichern und mit ```load()``` kann man so einen "json" Tokenizer in das Programm laden. Die ```tokenize()``` Funktion gibt dann nach einem Aufruf die Tokens des Inputs zurück.

Zusätzlich zu der Implementierung von Grund auf, befindet sich in ```bpe_tokenizer_with_import.py``` eine Implementierung mit Hilfe der Hugging Face "tokenizer" Bibliothek. Diese Bibliothek bietet eine effizientere und robustere Implementierung des BPE Tokenizers als die eigene Implementierung von Grund auf. Das Trainieren der Tokenizer hat auf meinem schon sehr alten Laptop ca. 10 Minuten gedauert, bzw. sogar über 60 Minuten bei dem bilingualen Corpus, was für solch kleine Datensätze ziemlich lang ist. Beim Nutzen der Hugging Face Bibliothek dauerte das Trainieren mit gleichem Corpus nur wenige Sekunden. Die trainierten Tokenizer in dem ```trained_tokenizers``` Ordner sind aus der Implementierung von Grund auf, aus ```bpe_tokenizer.py```. 

Als Datensatz habe ich die englische, deutschte und deutsch-englische Übersetzung der Bibel genutzt und den Datensatz vom Auswärtigen Amt auf Deutsch und Englisch. Wenn man sich die json Datei des Tokenizers anschaut fällt direkt auf, dass Bibel-spezifische Wörter, wie "Jesus" oder "Israel" und im Englischen "sin" oder "Lord" in dem Vokabular des trainierten Tokenizers auftauchen. Dies liegt daran, dann diese Wörter besonders oft im Corpus vorkommen und eben von dem BPE Algorithmus erkannt werden.

Einige Beispielsätze in den verschiedenen Tokenizern:

"Das ist ein Test" wird tokenisiert als:
- Deutscher Bibel Tokenizer: ['Das', 'ist', 'ein', 'T', 'es', 't'] (6)
- Englischer Bibel Tokenizer: ['D', 'as', 'i', 'st', 'e', 'in', 'T', 'est'] (8)
- Bilingualer Bibel Tokenizer: ['Da', 's', 'i', 'st', 'ein', 'T', 'est'] (7)
- Deutscher Amt Tokenizer: ['Da', 's', 'ist', 'ei', 'n', 'T', 'es', 't'] (8)
- Englischer Amt Tokenizer: ['D', 'as', 'ist', 'ei', 'n', 'T', 'est'] (7)

"Die Katze sitz auf dem Dach" wird tokenisiert als:
- Deutscher Bibel Tokenizer: ['Di', 'e', 'K', 'a', 'tz', 'e', 'si', 'tz', 'auf', 'de', 'm', 'Da', 'ch'] (13)
- Englischer Bibel Tokenizer: ['D', 'i', 'e', 'K', 'at', 'z', 'e', 'si', 't', 'z', 'au', 'f', 'de', 'm', 'D', 'ach'] (16)
- Bilingualer Bibel Tokenizer: ['Die', 'K', 'at', 'z', 'e', 'si', 'tz', 'auf', 'dem', 'Da', 'ch'] (11)
- Deutscher Amt Tokenizer:  ['Di', 'e', 'K', 'at', 'z', 'e', 'si', 'tz', 'auf', 'dem', 'Da', 'ch'] (12)
- Englischer Amt Tokenizer: ['D', 'i', 'e', 'K', 'at', 'z', 'e', 'si', 't', 'z', 'au', 'f', 'de', 'm', 'D', 'ach'] (16)

"The cat sits on the roof" wird tokenisiert als:
- Deutscher Bibel Tokenizer: ['T', 'h', 'e', 'c', 'a', 't', 'si', 'ts', 'on', 't', 'h', 'e', 'ro', 'o', 'f'] (15)
- Englischer Bibel Tokenizer: ['Th', 'e', 'ca', 't', 'si', 'ts', 'on', 'the', 'ro', 'of'] (10)
- Bilingualer Bibel Tokenizer: ['Th', 'e', 'ca', 't', 'si', 'ts', 'on', 'the', 'ro', 'of'] (10)
- Deutscher Amt Tokenizer: ['T', 'he', 'c', 'at', 'si', 'ts', 'on', 't', 'he', 'ro', 'o', 'f'] (12)
- Englischer Amt Tokenizer:  ['The', 'c', 'at', 'si', 'ts', 'on', 'the', 'ro', 'of'] (9)

"Jesus ist am Kreuz gestorben" wird tokenisiert als:
- Deutscher Bibel Tokenizer: ['Jesus', 'ist', 'am', 'K', 're', 'u', 'z', 'ges', 't', 'or', 'be', 'n'] (12)
- Englischer Bibel Tokenizer: ['Je', 'su', 's', 'i', 'st', 'am', 'K', 're', 'u', 'z', 'ge', 'st', 'or', 'be', 'n'] (15)
- Bilingualer Bibel Tokenizer: ['Jesus', 'i', 'st', 'am', 'K', 'reu', 'z', 'ge', 'st', 'or', 'be', 'n'] (12)
- Deutscher Amt Tokenizer: ['J', 'es', 'u', 's', 'ist', 'am', 'K', 're', 'u', 'z', 'ges', 't', 'or', 'be', 'n'] (15)
- Englischer Amt Tokenizer: ['J', 'es', 'us', 'ist', 'am', 'K', 're', 'u', 'z', 'ge', 'st', 'or', 'be', 'n'] (14)

"Jesus died on the cross" wird tokenisiert als:
- Deutscher Bibel Tokenizer: ['Jesus', 'di', 'e', 'd', 'on', 't', 'h', 'e', 'c', 'ro', 'ss'] (11)
- Englischer Bibel Tokenizer: ['Je', 'su', 's', 'di', 'ed', 'on', 'the', 'c', 'ro', 'ss'] (10)
- Bilingualer Bibel Tokenizer: ['Jesus', 'di', 'ed', 'on', 'the', 'c', 'ro', 'ss'] (8)
- Deutscher Amt Tokenizer: ['J', 'es', 'u', 's', 'di', 'e', 'd', 'on', 't', 'he', 'c', 'ro', 'ss'] (13)
- Englischer Amt Tokenizer: ['J', 'es', 'us', 'di', 'ed', 'on', 'the', 'c', 'ro', 'ss'] (10)


### Teilaufgabe c) (teilweise vorher schon beantwortet)
Ein grundlegendes Problem aller trainierten Tokenizer ist die starke Abhängigkeit vom Trainigsdatensatz. Die Bibeltexte enthalten archaische Sprache und einen begrenzten Wortschatz, was zu Problemen bei modernen Texten führt. Auch zeigt sich in den Beispielen, dass die Sprache eine wichtige Rolle spielt. Allgemein kann man sagen, dass die beste Repräsentation von Informationen von Tokenizern erreicht wird, die auf Daten trainiert wurden, die den Zieldaten in Sprache und Domäne am ähnlichsten sind. So ist ein bilingualer Tokenizer für gemischte Text vielleicht nützlich, aber verliert in einsprachigen Texten seine Effizienz und stellt somit eine Art Kompromiss dar.


## Aufgabe 2

Da die glove Datei zu groß ist für git, muss diese in den ```models_and_vocab``` Ordner heruntergeladen werden (https://nlp.stanford.edu/projects/glove/), bzw. muss eventuell der Pfad in Zeile 95 angepasst werden (```glove_model = load_glove_model("./models_and_vocab/glove.6B.50d.txt")```). Ich habe den "6B token Vektor" genutzt mit "50d". 
Beim Transformer Embedding dauert das finden der richtigen vocabs auf meinem Gerät sehr lange. Auch mit einer Begrenzung der vocabs auf 1000 Wörter in Zeile 77 (```vocab = vocab[:1000]```) dauert es nach der Eingabe einer Expression ca. 40 Sekunden bis ein Ergebnis auftaucht. Mit GloVe tauchen die Ergebnisse sofort auf.

### Beispielrechnung

#### Mit transformer:

Enter a word expression (E.g. king - man + woman): germany - berlin + paris
Results: 
- germany :  0.8296122550964355
- france :  0.7055253386497498
- canada :  0.6583239436149597
- golf :  0.6206454634666443
- texas :  0.6191905736923218

#### Mit GloVe:

Enter a word expression (E.g. king - man + woman): germany - berlin + paris
Results: 
- france :  0.9358007907867432
- paris :  0.8381515145301819
- french :  0.8359230756759644
- belgium :  0.8248729109764099
- germany :  0.8010828495025635

### Teilaufgabe b)
Die Unterschiede zwischen dem GloVe und dem Transformer Embedding sind unter anderem das nicht vorhandene Vokabular beim Transformer Embedding. Mit GloVe hat man direkt Zugriff auf das ganze Vokabular (trainiertes Vokabular). Mit BERT berechnet man für jedes Wort ein Embedding-Vektor und führt die Rechenoperationen im Vektorraum aus. Daraus erhält man ein Ergebnisvektor. Im Gegensatz zu GloVe kann man bei BERT nicht einfach ```model.similar_by_vector aufrufen``` aufrufen, weil BERT kein eingebautes Wörterbuch mit Vektoren für alle Wörter hat.
Für Transformer Embeddings muss auf externes Vokabular (Wörterbuch) zugegriffen werden. Das kann zum Beispiel das Vokabular vom Tokenizer sein oder ein eigenes, wie in diesem Fall (```google-10000-english.txt```). Das ist ein grundlegender Unterschied zwischen klassischen Embeddings (wie GloVe/Word2Vec) und Transformer-Modellen.

GloVe/Word2Vec:
- Haben ein fixes Vokabular und für jedes Wort einen Vektor.
- Man kann für jeden beliebigen Vektor die ähnlichsten Wörter im Vokabular finden (```similar_by_vector```).
Transformer (BERT etc.):
- Erzeugen Embeddings on the fly für beliebige Eingaben.
- Kein eingebautes Wörterbuch mit allen Wort-Vektoren.
- Um das ähnlichste Wort zu finden, muss man für eine Liste von Kandidaten (z.B. häufige Wörter oder dein eigenes Wörterbuch) die Embeddings berechnen und vergleichen (Cosinus-Ähnlichkeit).

Vorteile von traditionellem Embedding sind die Effizienz und die Einfachheit. Es ist ein geringer Rechenaufwand, da nach dem Trainieren nur ein einfacher Nachschlag nötig ist. Ein Nachteil ist das Fehlen von Kontextsensitivität. Ein Wort wie "Bank" hat immer den gleichen Vektor, egal ob Geldinstitut oder Sitzmöglichkeit, kann also Mehrdeutigkeit nicht auflösen.

Ein transformer-basiertes Embedding hat diese Kontextsensitivität und hat verschiedene Repräsentationen für dasselbe Wort in unterschiedlichen Kontexten. Erfasst weden sowohl syntaktische als auch semantische Beziehungen. Auch kann solch ein Embedding durch Fine-Tuning an eine spezifische Domain angepasst werden. Ein großer Nachteil ist natürlich die Komplexität und der hohe Rechenleistungsbedarf.

### Teilaufgabe c)
Rechenaufgaben wie Addition oder Subtraktion machen bei solchen semantischen und linguistischen Berechnungen Sinn. Operationen wie Multiplikation, Division oder Potenzierung haben keine klare semantisch-linguistische Interpretation und sind deshalb nicht sinvoll.


## Aufgabe 3

Hier steht was zu 3.


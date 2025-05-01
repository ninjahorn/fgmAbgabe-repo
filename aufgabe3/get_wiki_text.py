import wikipedia

wikipedia.set_lang('en')

try:
    article = wikipedia.page("Natural language processing").content
    
    with open("nlp_wiki_en.txt", "w", encoding="utf-8") as file:
        file.write(article)
    
    print("Article saved successfully to nlp_wiki_en.txt")
    
except Exception as e:
    print(f"Error: {e}")
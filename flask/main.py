import spacy
import wikipediaapi

def run_model():
    nlp = spacy.load("en_core_web_lg")
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    for token in doc:
        print(token.text, token.pos_, token.dep_)

def moovie_wiki():
    wiki_wiki = wikipediaapi.Wikipedia('MyMovieEval (example@example.com)', 'en')
    barbie = wiki_wiki.page('Barbie_(film)').summary
    oppenheimer = wiki_wiki.page('Oppenheimer_(film)').summary

    print(barbie)
    print()
    print(oppenheimer)

    nlp = spacy.load("en_core_web_lg")
    doc1 = nlp(barbie)
    doc2 = nlp(oppenheimer)
    doc1.similarity(doc2)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #run_model()
    moovie_wiki()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

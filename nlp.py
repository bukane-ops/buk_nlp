# import nltk
# from nltk.corpus import stopwords
# nltk.download("stopwords")

# removing stopword
# text = """This is a great example to demonstrate basic 
#        NLP tasks using NLTK library. and also stop words removal"""

# stopwords = set(stopwords.words('english'))
# print(f"stopwords: {stopwords}")

# remove stopwords from my text and save in a variable 
# words = text.lower().split()
# filtered_words = [word for word in words if word not in stopwords]
# print(f"filtered_words: {filtered_words}")

# 32 languages supported by nltk stopwords and 30 by spacy
# List all languages that have stopwords in NLTK
# languages = stopwords.fileids()

# Print them
# print("Languages supported by NLTK stopwords:\n")
# print(languages)

# Print total count
# print(f"\nTotal number of languages: {len(languages)}")



# stop wording for greek
# import nltk
# from nltk.corpus import stopwords
# nltk.download("stopwords")

# text = """τεχνητή νοημοσύνη αλλάζει τον κόσμο μας."""

# stopwords = set(stopwords.words("greek"))
# words = text.lower().split()

# filtered_words = [word for word in words if word not in stopwords]
# print(f"filtered: {filtered_words}")


# stemming vs lemmatization
# from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

# porterstemmer
# porter = PorterStemmer()
# words = ["running", "runner", "ran", "easily", "fairly", "fairness", "studies", "studying", "studied"]
# stemmed_word = [porter.stem(word) for word in words]
# print(f"porter: {stemmed_word}\n\n")

# snowballstemmer
# snow = SnowballStemmer('english')
# stemmed_word_snow = [snow.stem(word) for word in words]
# print(f"snow: {stemmed_word_snow}\n\n")
# to know number of languages supported
# print(SnowballStemmer.languages)

# lancasterstemmer
# lanc = LancasterStemmer()
# lanc_stemmed = [lanc.stem(word) for word in words]
# print(f"lanc: {lanc_stemmed}")

# Lemmatization
# import spacy
# spacy.cli.download("en_core_web_sm")

# includes
# """ Tokenizer
# Part-of-speech (POS) tagger
# Dependency parser
# Lemmatizer
# Named Entity Recognizer (NER) """

# import spacy
# nlp = spacy.load("en_core_web_sm")
# text = """Anthropods" is a misspelling of "arthropods," which are invertebrates with a hard exoskeleton, 
#         segmented bodies, and jointed appendages. They belong to the phylum Arthropoda and represent the largest
#           and most diverse phylum in the animal kingdom, inhabiting nearly every environment on Earth. Major groups
#             include insects, arachnids, crustaceans, and myriapods (centipedes and millipedes). """
# doc = nlp(text)
# lemmatized = [token.lemma_ for token in doc]
# print(lemmatized)

# import spacy
# nlp = spacy.load("en_core_web_sm")
# text = """A mammal (from Latin mamma 'breast')[1] is a vertebrate animal of the class Mammalia 
#  Mammals are characterised by the presence of milk-producing mammary glands for
#  feeding their young, a broad neocortex region of the brain, fur or hair, and three middle ear
#    bones. These characteristics distinguish them from reptiles and birds, from which their ancestors
#      diverged in the Carboniferous Period over 300 million years ago. Around 6,640 extant species of
#        mammals have been described and divided into 27 orders.[2] The study of mammals is called 
#        mammalogy.The largest orders of mammals, by number of species, are the rodents, bats, and
#          eulipotyphlans (including hedgehogs, moles and shrews). The next three are the primates
#            (including humans, monkeys and lemurs), the even-toed ungulates (including pigs, camels,
#              and whales), and the Carnivora (including cats, dogs, and seals)."""
# doc = nlp(text)
# for token in doc:
#     print(f"{token.text}, --->: {token.lemma_}")

# nlp = spacy.load("en_core_web_sm")
# text = "The Process was Slowing, but Helpful and Efficient!"
# doc = nlp(text)
# cleaned_text = []
# for token in doc:
#     if not token.is_stop and not token.is_punct:
#         cleaned_text.append(token.lemma_.lower())
# print(f"{cleaned_text}")

# tokenization
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download("punkt_tab") # download punkt_tab
# text = """The Process was Slowing, but Helpful and Efficient!"""
# tokens = word_tokenize(text)
# print(f"word Tokens: {tokens}")

# text = """so spacy handle all the lower case conversion? 
# but how do we now remove stopwords and punctuation. do i have to 
# use nltk for this and later use only spacy for lemma?"""

# tokens = sent_tokenize(text)
# print(f"word Tokens: {tokens}")

#Bag of words
# from sklearn.feature_extraction.text import CountVectorizer
# spam emails
# corpus = [
#     "Congratulations! You've won a free lottery ticket. Click here to claim your prize.",
#     "Dear user, your account has been compromised. Please reset your password immediately.",
#     "Limited time offer! Buy one get one free on all products. Don't miss out!",
#     "Hello friend, just wanted to check in and see how you're doing.",
#     "Reminder: Your appointment is scheduled for tomorrow at 10 AM."
# ]
# labels = ['spam','ham','spam','ham','ham']
# create the bag of words model
# vectorizer = CountVectorizer()
# bow_matrix = vectorizer.fit_transform(corpus)
# print('=='*40)
# print(f"vocabulary: {vectorizer.get_feature_names_out()}")
# print('=='*40)
# print('\nBag of words matrix for each email:\n', bow_matrix.toarray())
# print('=='*40)

# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = ["the cat sat on the mat",
             "the dog sat on the log",
             "cats and dogs are animals"]

# matrix creation is our goal
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

df = pd.DataFrame(tfidf_matrix.toarray(),
                  columns = tfidf_vectorizer.get_feature_names_out(),
                  index = ['Doc1','Doc2','Doc3'])
print(df.round(2))

# multivariant

# print(f"doc: {tfidf_vectorizer.get_feature_names_out()}")
# print('tfdif :\n', tfidf_matrix.toarray())

# https://docs.google.com/presentation/d/1yRGh4CAE3BEI27MpEGEMMEJ7XgOl25I4LtnsW-W3qCY/edit?usp=sharing
"""Customer Feedback Analyser using NLP techniques.

This module provides comprehensive analysis of customer feedback including
sentiment analysis, entity extraction, and topic modeling.
"""
# necessary libraries
# python -m spacy download en_core_web_lg # need this to load large English NLP model, please amend syntax if running on vs code i.e remove !
import spacy # for NER
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer # text to numbers
from sklearn.decomposition import LatentDirichletAllocation # for topic modeling
from transformers import pipeline # for sentiment analysis
from collections import Counter


class FeedbackAnalyser: # the blueprint of the program
    """A comprehensive customer feedback analyzer using NLP techniques.

    This class provides methods for sentiment analysis, entity extraction,
    and topic modeling of customer reviews and feedback.
    """

    def __init__(self): # defines method inside the class ( constructor method)
      # self initialize everything that the class will need, load heavy resources once, not everytime a method is called
      # self is how each object keeps track of its own variables and methods
        """Initialize the FeedbackAnalyzer with required models.

        Loads the sentiment analysis pipeline from transformers and
        the spaCy language model for entity recognition.
        """
        print(f"Loading Models...(this may take a while)")
        self.sentiment_analyser = pipeline("sentiment-analysis") # sentiment model; designed to handle multiple input - result is going to be a list of dictionaries (batch model)
        self.nlp = spacy.load("en_core_web_lg") # pretrained large english nlp model ; tokenization,pos tagging, ner
        # both self.sentiment_analyzer = pipeline("sentiment-analysis") and self.nlp = spacy.load("en_core_web_lg") are heavy models and both are pretrained.
        # reloading everytime will cost time wasting and memory
        # topic modeling was not loaded; unlike both ner and sentiment analysis, topic modeling isn't pretrained
        # it is unsupervised, it learns topics from the specific dataset, every new set of customer reviews may have completely different themes
        # so topic model can't be loaded in advanced, it is trained on input text each time
        print(f"Models Loaded Successfully.")

    def analyse_sentiment(self, reviews): # reviews, the parameter expected to be a list of strings, where each string is one customer review
      # self allows you to use the variables and other things in the constructor method
        """Analyze sentiment of customer reviews.

        Args:
            reviews (list): List of review strings to analyze.

        Returns:
            list: List of dictionaries containing review, label, and confidence.
        """
        results = []
        for review in reviews:
            sentiment = self.sentiment_analyser(review)[0] # runs the model on the text and returns a list of dictionaries
            #[0] grabs the first(and only) result from the list, sentiment will now hold a dictionary i.e label and score, this is the code saying [0]
            # I only gave you one review, so take the first(and only) element of the list
            results.append({ # adds a new dictionary to the list results
                "review": review, # stores the original review text
                "label": sentiment['label'], # takes the label (POSITIVE / NEGATIVE) from the model output
                "confidence": round(sentiment['score'],2) # rounds the numeric score (e.g. 0.987654 → 0.99)
            })
        return results # This sends the final list of sentiment results back to whoever called this method

    def extract_entities(self, reviews): # function to scan products, people, organizations and places, groups and counts them
      # purpose: Go through all reviews, identify key entities (products, people, organizations, etc.), and return the most frequently mentioned ones
        """Extract named entities from customer reviews.

        Args:
            reviews (list): List of review strings to process.

        Returns:
            dict: Dictionary with entity types as keys and most common entities as values.
        """
        all_entities = {
            "PRODUCT":[],
            "ORG":[],
            "GPE":[],
            "PERSON":[]
        } # initiasing the entity container
        for review in reviews:
            doc = self.nlp(review) # full nlp analysis object representing my text; contains Tokens, POS tags, lemmas, syntactic relations, entities, sentences, noun chunks, embeddings attributes(doc.sents, doc.ents, doc.noun_chunks) - dir(doc) - spacy doc object
            for ent in doc.ents:
                if ent.label_ in all_entities: # ent has attributes like text, label_
                    all_entities[ent.label_].append(ent.text) # and the label_ is found in my all_entities variable
        entity_counts = {}
        for ent_type, entities in all_entities.items(): # iterates through the dictionary as (key, value) pairs
            if entities: # skip types that have empty lists (no matches found)
                entity_counts[ent_type] = Counter(entities).most_common(5) # counts how many times each entity appears and returns the top 5 most requent entities for the category as a list of tuples
        return entity_counts

    def discover_topics(self, reviews, num_topics=3): # topic modeling; discovers themes or topics hidden in our customer reviews
      # num_topics, default parameter; if no value is passed, it will find 3 topics
        """Discover topics in customer reviews using LDA.

        Args:
            reviews (list): List of review strings to analyze.
            num_topics (int): Number of topics to discover. Default is 3.

        Returns:
            list: List of dictionaries containing topic numbers and keywords.
        """
        vectorizer = CountVectorizer(stop_words='english') # convert text into numbers i.e document_term matrix; stop_words removes common English filler words( like the, is, and; they aren't helpful for discovering topics)
        try:
            doc_matrix = vectorizer.fit_transform(reviews) # fit: Learn all the unique words (the vocabulary); transform: Convert each review into a vector of word counts. result is a sparse matrix of shape (n_documents, n_words)
            lda = LatentDirichletAllocation(n_components = num_topics,
                                            random_state=42,max_iter=10) # number of passes over the data during training
            lda.fit(doc_matrix) # trains the lda model on the document-term matrix
            words = vectorizer.get_feature_names_out() # extracts the list of all the words (columns) the vectorizer learned
            topics = []
            for topic_idx, topic in enumerate(lda.components_): # each word corresponds to a column index in lda.components_, a NumPy array of shape (n_topics, n_words)
                # topic_idx; topic number (starting from 0) and topic; the array of word weights for that topic
                top_words_idx = topic.argsort()[-5:][::-1] # returns indices that would sort the topic weights in ascending order, and takes the last 5 indices(the largest weights),and reverses the order to get them in descending
                # order (most important first)
                top_words = [words[i] for i in top_words_idx] # for each index in top_words_idx, pick the actual word from the words list
                topics.append({
                    "topic_number": topic_idx + 1,
                    "keywords": top_words})
            return topics
        except Exception as e:# try...except block ensures that if something goes wrong (e.g., empty data, model training error), the program won't crash
            print(f"Error in topic discovery: {e}")

    def get_summary_stats(self, sentiment_results): # this is the list returned by the analyse_sentiment() method.
        """Calculate summary statistics from sentiment analysis results.

        Args:
            sentiment_results (list): List of sentiment analysis results.

        Returns:
            dict: Dictionary containing review counts and percentages.
        """
        sentiments = [r['label'] for r in sentiment_results] # It loops over each result r in the sentiment_results list and pulls out only the 'label' value
        total = len(sentiments)
        positive = sentiments.count('POSITIVE')
        negative = sentiments.count('NEGATIVE')
        return {
            "total_reviews": total,
            "positive_reviews": positive,
            "negative_reviews": negative,
            "positive_percentage": round(positive / total * 100, 2),
            "negative_percentage": round(negative / total * 100, 2)
        } # returns a dictionary summarizing the results in a nice structured form

    def analyse_all(self, reviews): # acts as the control center that brings everything together — sentiment analysis, entity extraction, and topic modeling — into one clean, end-to-end pipeline
        """Perform comprehensive analysis of customer reviews.

        Args:
            reviews (list): List of review strings to analyze.

        Returns:
            dict: Complete analysis results including sentiment, entities, and topics.
        """
        print("=="*50)
        print(f"CUSTOMER FEEDBACK ANALYSIS REPORT")
        print("=="*50)

        #1. Sentiment Analysis
        print("\n1. Sentiment Analysis")
        sentiment_results = self.analyse_sentiment(reviews) # Runs the Hugging Face model on every review and returns a list of dictionaries like [{"review": "Great phone!", "label": "POSITIVE", "confidence": 0.98}, ...]
        stats = self.get_summary_stats(sentiment_results) # passes the raw sentiment data to the summarizer function and returns an aggregated stats dictionary

        #2. Entity Extraction
        print("\n2. Entity Extraction")
        entities = self.extract_entities(reviews) # Uses spaCy to find named entities in each review (e.g., product names, places, organizations)

        #3. Topic Discovery - topic modelling
        print("\n3. Topic Discovery")
        topics = self.discover_topics(reviews) # Uses CountVectorizer to turn text into word counts and Uses LatentDirichletAllocation to learn hidden discussion topics

        print(f"Analysis Complete!.\n")

        return {
            "sentiment_results": sentiment_results,
            "stats": stats,
            "entities": entities,
            "topics": topics
        } # returns a single dictionary containing everything my analysis generated; turns to be the results parameter passed in the function print_results


def print_results(results): # the input parameter; expected to be the dictionary, the reporting layer of the feedback analyser; standalone and not part of the class
    """Print formatted analysis results.

    Args:
        results (dict): Analysis results from analyse_all method.
    """
    print("=="*50)
    print("Summary Statistics:")
    stats = results['stats']
    print(f"Total Reviews: {stats['total_reviews']}")
    print(f"Positive Reviews: {stats['positive_reviews']} ({stats['positive_percentage']}%)")
    print(f"Negative Reviews: {stats['negative_reviews']} ({stats['negative_percentage']}%)") # all from get_summary_stats method

    #Sentiment Details
    print("\n" + "=="*20 + " Individual review Sentiments " + "=="*20)
    for i, result in enumerate(results['sentiment_results'][:5],1): # loop through the first five sentiment results while also keeping a counter starting at 1
        sentiment_emoji = "😊" if result['label'] == 'POSITIVE' else "😞" # If the sentiment label is POSITIVE, show a smiling face; otherwise, a sad face
        print(f"\n{i}. Review: {result['review']}\n   Sentiment: {result['label']} {sentiment_emoji} (Confidence: {result['confidence']})") # [{"review": "Great phone!", "label": "POSITIVE", "confidence": 0.98}, ...]

    if len(results['sentiment_results']) > 5:
        print(f"\n... and {len(results['sentiment_results']) - 5} more reviews analyzed.") # It prevents flooding the output with too many lines

    #Topics Details
    print("\n" + "=="*20 + " Topic Discovery " + "=="*20)
    for topic in results['topics']:
        print(f"\nTopic {topic['topic_number']}: " + ", ".join(topic['keywords'])) # joins the list of keywords into a single comma-separated string

    #Entities Details
    print("\n" + "=="*20 + " Extracted Entities " + "=="*20)
    entity_labels = {
        "PRODUCT":"Products",
        "ORG":"Organisations",
        "GPE":"Locations",
        "PERSON":"People mentioned"
    }
    entities = results['entities']
    if entities: # Checks if the entities dictionary is not empty
        for ent_type, labels in entity_labels.items():
            if ent_type in entities:
                print(f"\n{labels}:")
                for entity, count in entities[ent_type]:
                    print(f" - {entity} (mentioned {count} times)")
    else:
        print("No significant entities found.")

if __name__ == "__main__": # acts as a launch button; If this file is the one being run directly (not imported),then execute the following code.
    """Example usage of the FeedbackAnalyzer class."""
    # Sample customer reviews for demonstration

    sample_reviews = [
    "I absolutely love the new iPhone 15! The camera quality is insane and the battery lasts forever.",
    "The customer support at Samsung was terrible. I waited on hold for 45 minutes and got no help.",
    "This laptop from Dell performs great for my daily work, but it overheats sometimes.",
    "I'm really disappointed with the headphones. The sound quality is poor and they feel cheap.",
    "Amazon delivery was super fast, and the packaging was perfect. I'm very satisfied with the service!",
    "The restaurant's pasta was amazing, but the service was slow and the waiter seemed rude.",
    "The new Windows 11 update made my PC slower and some apps crash frequently.",
    "Had a fantastic experience at the Hilton hotel. The room was clean and the staff were very professional.",
    "The Nike running shoes are comfortable but not durable — they tore after a month.",
    "Google's new Pixel phone is impressive! Smooth performance and the camera beats the competition." 
   ]

    analyser = FeedbackAnalyser()
    analysis_results = analyser.analyse_all(sample_reviews)
    print_results(analysis_results)
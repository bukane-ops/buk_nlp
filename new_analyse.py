"""Customer Feedback Analyser using Advanced NLP Techniques.

This module provides comprehensive analysis of customer feedback including
sentiment analysis, entity extraction, and topic modelling using BERTopic.
"""
# Necessary libraries
import spacy
import warnings
warnings.filterwarnings("ignore")
from bertopic import BERTopic
from transformers import pipeline
from collections import Counter
import logging

# Configure logging for better error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackAnalyser:
    """A comprehensive customer feedback analyser using advanced NLP techniques.

    This class provides methods for sentiment analysis, entity extraction,
    and topic modelling of customer reviews and feedback with optimised
    batch processing for large datasets.
    """

    def __init__(self, sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                 spacy_model="en_core_web_lg", batch_size=32):
        """Initialise the FeedbackAnalyser with required models.

        Args:
            sentiment_model (str): HuggingFace model for sentiment analysis
            spacy_model (str): spaCy model for entity recognition
            batch_size (int): Batch size for processing large datasets
        """
        logger.info("Loading models... (this may take a while)")
        
        # Initialise sentiment analyser with batch processing capability
        # This model handles POSITIVE, NEGATIVE, and NEUTRAL sentiments
        self.sentiment_analyser = pipeline(
            "sentiment-analysis", 
            model=sentiment_model,
            return_all_scores=True  # Returns all sentiment scores including neutral
        )
        
        # Load spaCy model for entity recognition
        self.nlp = spacy.load(spacy_model)
        
        # Set batch size for optimised processing
        self.batch_size = batch_size
        
        # Initialise BERTopic model (will be created when needed)
        self.topic_model = None
        
        logger.info("Models loaded successfully.")

    def analyse_sentiment(self, reviews):
        """Analyse sentiment of customer reviews using batch processing.
        
        Handles POSITIVE, NEGATIVE, and NEUTRAL sentiments with confidence scores.
        Optimised for large datasets using batch processing.

        Args:
            reviews (list): List of review strings to analyse

        Returns:
            list: List of dictionaries containing review, label, and confidence
        """
        if not reviews:
            logger.warning("No reviews provided for sentiment analysis")
            return []
        
        try:
            # Process reviews in batches for better performance
            results = []
            
            # The pipeline can handle batch processing internally
            batch_results = self.sentiment_analyser(reviews)
            
            for i, review in enumerate(reviews):
                # Get the sentiment with highest confidence
                sentiment_scores = batch_results[i]
                best_sentiment = max(sentiment_scores, key=lambda x: x['score'])
                
                # Map model labels to standardised labels
                label_mapping = {
                    'LABEL_0': 'NEGATIVE',
                    'LABEL_1': 'NEUTRAL', 
                    'LABEL_2': 'POSITIVE'
                }
                
                standardised_label = label_mapping.get(
                    best_sentiment['label'], 
                    best_sentiment['label']
                )
                
                results.append({
                    "review": review,
                    "label": standardised_label,
                    "confidence": round(best_sentiment['score'], 3)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Return empty results with error indication
            return [{"review": review, "label": "ERROR", "confidence": 0.0} 
                   for review in reviews]

    def extract_entities(self, reviews):
        """Extract named entities from customer reviews using batch processing.
        
        Uses spaCy's nlp.pipe for optimised batch processing of large datasets.
        Extracts products, organisations, locations, and people mentioned.

        Args:
            reviews (list): List of review strings to process

        Returns:
            dict: Dictionary with entity types as keys and most common entities as values
        """
        if not reviews:
            logger.warning("No reviews provided for entity extraction")
            return {}
        
        try:
            all_entities = {
                "PRODUCT": [],
                "ORG": [],
                "GPE": [],  # Geopolitical entities (locations)
                "PERSON": []
            }
            
            # Use spaCy's nlp.pipe for efficient batch processing
            # This is much faster than processing reviews one by one
            for doc in self.nlp.pipe(reviews, batch_size=self.batch_size):
                for ent in doc.ents:
                    if ent.label_ in all_entities:
                        # Clean and normalise entity text
                        entity_text = ent.text.strip()
                        if len(entity_text) > 1:  # Filter out single characters
                            all_entities[ent.label_].append(entity_text)
            
            # Count and return most common entities
            entity_counts = {}
            for ent_type, entities in all_entities.items():
                if entities:
                    # Get top 5 most frequent entities for each category
                    entity_counts[ent_type] = Counter(entities).most_common(5)
            
            return entity_counts
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {}

    def discover_topics(self, reviews, num_topics=None, min_topic_size=2):
        """Discover topics in customer reviews using BERTopic.
        
        BERTopic provides more accurate and contextual topic modelling
        compared to traditional LDA approaches.

        Args:
            reviews (list): List of review strings to analyse
            num_topics (int, optional): Number of topics to discover. 
                                      If None, BERTopic will determine automatically
            min_topic_size (int): Minimum number of documents per topic

        Returns:
            list: List of dictionaries containing topic numbers and keywords
        """
        if not reviews or len(reviews) < min_topic_size:
            logger.warning(
                f"Insufficient reviews for topic discovery. "
                f"Need at least {min_topic_size} reviews, got {len(reviews)}"
            )
            return []
        
        try:
            # Initialise BERTopic model with configuration
            self.topic_model = BERTopic(
                nr_topics=num_topics,
                min_topic_size=min_topic_size,
                calculate_probabilities=True,
                verbose=False
            )
            
            # Fit the model and get topics
            topics, probabilities = self.topic_model.fit_transform(reviews)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            
            # Format results
            formatted_topics = []
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                
                # Skip outlier topic (usually topic -1)
                if topic_id == -1:
                    continue
                
                # Get top words for this topic
                topic_words = self.topic_model.get_topic(topic_id)
                
                if topic_words:
                    # Extract just the words (not the scores)
                    keywords = [word for word, _ in topic_words[:5]]
                    
                    formatted_topics.append({
                        "topic_number": topic_id + 1,  # Make it 1-indexed for display
                        "keywords": keywords,
                        "document_count": row['Count']
                    })
            
            return formatted_topics
            
        except Exception as e:
            logger.error(f"Error in topic discovery: {e}")
            
            # Fallback: return basic topic structure
            return [{
                "topic_number": 1,
                "keywords": ["error", "processing", "failed"],
                "document_count": 0
            }]

    def get_summary_stats(self, sentiment_results):
        """Calculate summary statistics from sentiment analysis results.
        
        Handles POSITIVE, NEGATIVE, and NEUTRAL sentiments with proper
        percentage calculations.

        Args:
            sentiment_results (list): List of sentiment analysis results

        Returns:
            dict: Dictionary containing review counts and percentages
        """
        if not sentiment_results:
            return {
                "total_reviews": 0,
                "positive_reviews": 0,
                "negative_reviews": 0,
                "neutral_reviews": 0,
                "positive_percentage": 0.0,
                "negative_percentage": 0.0,
                "neutral_percentage": 0.0
            }
        
        sentiments = [r['label'] for r in sentiment_results]
        total = len(sentiments)
        
        positive = sentiments.count('POSITIVE')
        negative = sentiments.count('NEGATIVE')
        neutral = sentiments.count('NEUTRAL')
        
        return {
            "total_reviews": total,
            "positive_reviews": positive,
            "negative_reviews": negative,
            "neutral_reviews": neutral,
            "positive_percentage": round(positive / total * 100, 2) if total > 0 else 0.0,
            "negative_percentage": round(negative / total * 100, 2) if total > 0 else 0.0,
            "neutral_percentage": round(neutral / total * 100, 2) if total > 0 else 0.0
        }

    def analyse_all(self, reviews):
        """Perform comprehensive analysis of customer reviews.
        
        Acts as the control centre that brings together sentiment analysis,
        entity extraction, and topic modelling into one end-to-end pipeline.

        Args:
            reviews (list): List of review strings to analyse

        Returns:
            dict: Complete analysis results including sentiment, entities, and topics
        """
        if not reviews:
            logger.error("No reviews provided for analysis")
            return {
                "sentiment_results": [],
                "stats": self.get_summary_stats([]),
                "entities": {},
                "topics": []
            }
        
        print("=" * 100)
        print("CUSTOMER FEEDBACK ANALYSIS REPORT")
        print("=" * 100)

        # 1. Sentiment Analysis
        print("\n1. Sentiment Analysis")
        sentiment_results = self.analyse_sentiment(reviews)
        stats = self.get_summary_stats(sentiment_results)

        # 2. Entity Extraction
        print("\n2. Entity Extraction")
        entities = self.extract_entities(reviews)

        # 3. Topic Discovery
        print("\n3. Topic Discovery")
        topics = self.discover_topics(reviews)

        print("Analysis complete!\n")

        return {
            "sentiment_results": sentiment_results,
            "stats": stats,
            "entities": entities,
            "topics": topics
        }


def print_results(results):
    """Print formatted analysis results.
    
    The reporting layer of the feedback analyser that displays results
    in a user-friendly format with British English spelling.

    Args:
        results (dict): Analysis results from analyse_all method
    """
    print("=" * 100)
    print("Summary Statistics:")
    stats = results['stats']
    print(f"Total Reviews: {stats['total_reviews']}")
    print(f"Positive Reviews: {stats['positive_reviews']} ({stats['positive_percentage']}%)")
    print(f"Negative Reviews: {stats['negative_reviews']} ({stats['negative_percentage']}%)")
    print(f"Neutral Reviews: {stats['neutral_reviews']} ({stats['neutral_percentage']}%)")

    # Sentiment Details
    print("\n" + "=" * 40 + " Individual Review Sentiments " + "=" * 40)
    for i, result in enumerate(results['sentiment_results'][:5], 1):
        # Handle different sentiment types with appropriate emojis
        emoji_map = {
            'POSITIVE': '😊',
            'NEGATIVE': '😞', 
            'NEUTRAL': '😐',
            'ERROR': '❌'
        }
        sentiment_emoji = emoji_map.get(result['label'], '❓')
        
        print(f"\n{i}. Review: {result['review']}")
        print(f"   Sentiment: {result['label']} {sentiment_emoji} "
              f"(Confidence: {result['confidence']})")

    if len(results['sentiment_results']) > 5:
        remaining = len(results['sentiment_results']) - 5
        print(f"\n... and {remaining} more reviews analysed.")

    # Topics Details
    print("\n" + "=" * 40 + " Topic Discovery " + "=" * 40)
    if results['topics']:
        for topic in results['topics']:
            keywords_str = ", ".join(topic['keywords'])
            doc_count = topic.get('document_count', 'N/A')
            print(f"\nTopic {topic['topic_number']}: {keywords_str}")
            print(f"   Documents: {doc_count}")
    else:
        print("No topics discovered.")

    # Entities Details
    print("\n" + "=" * 40 + " Extracted Entities " + "=" * 40)
    entity_labels = {
        "PRODUCT": "Products",
        "ORG": "Organisations",
        "GPE": "Locations",
        "PERSON": "People Mentioned"
    }
    
    entities = results['entities']
    if entities:
        for ent_type, labels in entity_labels.items():
            if ent_type in entities:
                print(f"\n{labels}:")
                for entity, count in entities[ent_type]:
                    print(f" - {entity} (mentioned {count} times)")
    else:
        print("No significant entities found.")


if __name__ == "__main__":
    """Example usage of the FeedbackAnalyser class with edge cases."""
    
    # Sample customer reviews including edge cases
    sample_reviews = [
        "I absolutely love the new iPhone 15! The camera quality is brilliant and the battery lasts forever.",
        "The customer support at Samsung was terrible. I waited on hold for 45 minutes and got no help.",
        "This laptop from Dell performs adequately for my daily work, but it overheats sometimes.",
        "I'm really disappointed with the headphones. The sound quality is poor and they feel cheap.",
        "Amazon delivery was super fast, and the packaging was perfect. I'm very satisfied with the service!",
        "The restaurant's pasta was amazing, but the service was slow and the waiter seemed rude.",
        "The new Windows 11 update made my PC slower and some apps crash frequently.",
        "Had a fantastic experience at the Hilton hotel. The room was clean and the staff were very professional.",
        "The Nike running shoes are comfortable but not durable — they tore after a month.",
        "Google's new Pixel phone is impressive! Smooth performance and the camera beats the competition.",
        # Edge cases
        "",  # Empty review
        "OK",  # Very short review
        "The product is fine I suppose nothing special really just average quality for the price point.",  # Neutral sentiment
        "😊😊😊 Love it! 💯",  # Emojis and informal text
        "AMAZING PRODUCT!!! HIGHLY RECOMMEND!!! 5 STARS!!!",  # All caps with punctuation
    ]
    
    # Filter out empty reviews for demonstration
    filtered_reviews = [review for review in sample_reviews if review.strip()]
    
    analyser = FeedbackAnalyser()
    analysis_results = analyser.analyse_all(filtered_reviews)
    print_results(analysis_results)
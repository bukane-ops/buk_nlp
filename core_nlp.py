from transformers import pipeline
def sentiment_analysis_demo(text):
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)

    # text = [
    #     'I love this movie! it is absolutely fantastic.',
    #     'This is the worst product I have ever bought.',
    #     'The weather today is I think ok but not ok but in a way ok!'
    # ]
    print(result)
    return result

sa = sentime_analysis_demo('I love natural language processing')
print(sa)
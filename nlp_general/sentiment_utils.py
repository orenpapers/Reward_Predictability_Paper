# from flair.data import Sentence
# from flair.models import TextClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelWithLMHead,  AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

# from textblob import TextBlob
from scipy.special import softmax
import nltk
import numpy as np
import urllib, csv

# nltk.download('vader_lexicon')
vader_analyser = SentimentIntensityAnalyzer()
# flair_sentiment = TextClassifier.load('en-sentiment')
try:
    t5_emotion_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    t5_emotion_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
except Exception as e:
    print("Cant fetch t5_emotion_tokenizer, model - ", e)
#
# emotion_MODEL = "cardiffnlp/twitter-roberta-base-emotion"
# emotion_MODEL = "/Users/orenkobo/Desktop/PhD/Aim2_new/HP_code/cardiffnlp_emotion_model/twitter-roberta-base-emotion"
# emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_MODEL)
# emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_MODEL)
# mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/mapping.txt"
# with urllib.request.urlopen(mapping_link) as f:
#     emotion_html = f.read().decode('utf-8').split("\n")
#     emotion_csvreader = csv.reader(emotion_html, delimiter='\t')
# emotion_labels = [row[1] for row in emotion_csvreader if len(row) > 1]


def extract_sentiment(segment):
    #https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair

    segment = str(segment)
    vd = get_vader_sentiment(segment) #['neg','neu','pos','compound']
    tb = get_textblob_sentiment(segment) #[polarity , subjectivity]
    tr = get_transformers_sentiments(segment) #[positive, negative[
    t5 = get_t5(segment) #label
    rb = get_emotion(segment) #['anger', 'joy', 'optimism', 'sadness']
    return  vd + tb  + tr + t5 + rb



def get_transformers_sentiments(word):
    #https://towardsdatascience.com/sentiment-analysis-with-pretrained-transformers-using-pytorch-420bbc1a48cd
    nlp = pipeline("sentiment-analysis")
    label = nlp(word)[0]['label']
    score = nlp(word)[0]['score']
    if label == "POSITIVE":
        return [score, 1-score]
    else:
        return [1-score, score]

def get_textblob_sentiment(word):
    textblob_score = [TextBlob(word).sentiment.polarity, TextBlob(word).sentiment.subjectivity]
    return textblob_score

# def get_flair_sentiment(word):
#     s = Sentence(word)
#     flair_score = [flair_sentiment.predict(s)] #todo fix, why none?
#     return flair_score

def get_vader_sentiment(word):
    vader_score = list(vader_analyser.polarity_scores(word).values())
    return vader_score


def get_t5(text):
    #https://huggingface.co/mrm8488/t5-base-finetuned-emotion

    input_ids = t5_emotion_tokenizer.encode(text + '</s>', return_tensors='pt')

    output = t5_emotion_model.generate(input_ids=input_ids,
                                       max_length=2)

    dec = [t5_emotion_tokenizer.decode(ids) for ids in output]
    if len(dec) > 1:
        a = 2
    label = dec[0].split("<pad> ")[1]
    # print("For text {} emotion is {} (dec is {})".format(text, label, dec))

    return [label]

def get_emotion(text):
    # https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion

    encoded_input = emotion_tokenizer(text, return_tensors='pt')
    output = emotion_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    for i in range(scores.shape[0]):
        l = emotion_labels[ranking[i]]
        s = scores[ranking[i]]
        # print(f"{i+1}) {l} {np.round(float(s), 4)}")
    return list(scores) #4 labels, ['anger', 'joy', 'optimism', 'sadness']



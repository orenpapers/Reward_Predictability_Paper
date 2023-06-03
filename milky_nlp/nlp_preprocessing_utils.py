from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string,re
# from gensim.utils import tokenize
from nltk.tokenize import word_tokenize
#from keras.preprocessing.text import text_to_word_sequence

SEGEMENT_DELIMETER = "."
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
# nlp = English()


def split_text_to_segments(text):
    list_of_segments = text.split(SEGEMENT_DELIMETER)
    return list_of_segments


def clean_text(segment, keep_dot):

    if keep_dot:
        to_remove = [x for x in string.punctuation if x != '.']
    else:
        to_remove = string.punctuation
    for c in to_remove:
        segment = segment.replace(c, "")
    return segment

def tokenize_text(text, method):
    # https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/
    if method=='nltk':
        word_tokens = word_tokenize(text)
        tokens = [w for w in word_tokens if not w in stop_words]

    if method=='spacy':
        my_doc = nlp(text)
        tokens = []
        for token in my_doc:
            tokens.append(token.text)

    if method=='re':
        tokens = re.findall("[\w']+", text)

    if method=='space':
        tokens = text.split(" ")

    if method=='keras':
        tokens = text_to_word_sequence(text)

    if method == 'gensim':
        tokens = list(tokenize(text))

    return tokens

def run_lemmatization(segment):
    # Examples of lemmatization: rocks : rock,  corpora : corpus, better : good
    return [lemmatizer.lemmatize(w) for w in segment]

def run_stemmization(segment):
    # Examples of stemming:  "likes", , "liked", "likely", "liking" -> "like"
    return [stemmer.stem(w) for w in segment]


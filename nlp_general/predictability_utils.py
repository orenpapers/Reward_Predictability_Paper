# from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel, pipeline
from transformers import BertTokenizer, BertLMHeadModel, BertConfig, AutoTokenizer, AutoModelForSequenceClassification
# from wordfreq import word_frequency, zipf_frequency
from happytransformer import HappyWordPrediction
import numpy as np
import torch, math

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model.eval()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertLMHeadModel.from_pretrained('bert-base-cased')
bert_model.eval()

happy_wp = HappyWordPrediction("BERT", "bert-base-uncased")


def extract_word_predicatbility(words, word_idx, word):
    if word_idx < 10:
        return [np.nan, np.nan, np.nan]
    words_back_5   = " ".join(words[word_idx - 5 : word_idx])
    words_back_10  = " ".join(words[word_idx - 10 : word_idx])
    words_back_all = " ".join(words[ : word_idx])
    return [get_word_masked_predictability(context = words_back_5 , word= word),
            get_word_masked_predictability(context = words_back_10 ,word= word),
            get_word_masked_predictability(context = words_back_all , word=word)]

def extract_sequence_predictability(sentence_text, take_hp = False):
    ret = [get_sequence_perplexity(sentence_text, model="gpt2"),
     get_sequence_perplexity(sentence_text, model="bert")
     ]
    if take_hp:
        try:
            ret.append([happy_wp.predict_mask(" ".join(sentence_text.split(" ")[:i]) + " [MASK]", targets=[w])[0].score
                for i,w in enumerate(sentence_text.split(" "))][2:])
        except Exception as e:
            ret.append([np.nan])
    return ret

def get_word_masked_predictability(context, word):
    result = happy_wp.predict_mask(context + " [MASK]", targets =[word])[0].score
    return result

def get_word_frequency(w, method):
    if method == "zipf":
        return zipf_frequency(w, "en")
    if method == "freq":
        return word_frequency(w, "en")


def get_sequence_perplexity(s, model):

    if model == 'gpt2':
        tokenizer = gpt2_tokenizer
        model = gpt2_model

    if model == 'bert':
        tokenizer = bert_tokenizer
        model = bert_model
    try:
        s = score(tokenizer, model, s)
    except Exception as e:
        print("Error while get perplexity for: {} - {}".format(s,e))
        return np.nan
    return s

def score(tokenizer, model, sentence):
    #https://huggingface.co/transformers/usage.html#language-modeling
    #https://github.com/google-research/bert/issues/323
    #https://github.com/huggingface/transformers/issues/37

    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss=model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())
import gensim.downloader as api
import torch
import math
from wordfreq import word_frequency, zipf_frequency
word_vectors_glove = api.load("glove-wiki-gigaword-100")
word_vectors_w2v = api.load('word2vec-google-news-300')

def get_glove(w):
    return word_vectors_glove[w]

def get_w2v(w):
    return word_vectors_w2v[w]

def word_freq(w):
    return word_frequency(w, 'en')

def word_zipf_freq(w):
    return zipf_frequency(w, 'en')

def word_in_sentence_prob(self, sentence, word):

    sequence = f"{sentence} {bert_tokenizer.mask_token}"

    input_ids = bert_tokenizer.encode(sequence, bert_tokenizer="pt")
    mask_token_index = torch.where(input_ids == bert_tokenizer.mask_token_id)[1]

    token_logits = bert_model(input_ids)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    mask_token_logits = torch.softmax(mask_token_logits, dim=1)

    top_5 = torch.topk(mask_token_logits, 5, dim=1)
    top_5_tokens = zip(top_5.indices[0].tolist(), top_5.values[0].tolist())

    for token, score in top_5_tokens:
        print(sequence.replace(bert_tokenizer.mask_token, bert_tokenizer.decode([token])), f"(score: {score})")

    # Get the score of token_id
    sought_after_token = word
    sought_after_token_id = bert_tokenizer.encode(sought_after_token, add_special_tokens=False, add_prefix_space=True)[
        0]  # 928

    token_score = mask_token_logits[:, sought_after_token_id]
    print(f"Score of {sought_after_token}: {mask_token_logits[:, sought_after_token_id]}")
    return token_score
from transformers import BertTokenizer, BertLMHeadModel, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch

def score(tokenizer, model, sentence, score_type):
    #https://huggingface.co/transformers/usage.html#language-modeling
    #https://github.com/google-research/bert/issues/323
    #https://github.com/huggingface/transformers/issues/37

    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss=model(tensor_input, labels=tensor_input)[0]
    if score_type == 'entropy':
        return loss.detach().numpy()
    if score_type == 'perplexity':
        return np.exp(loss.detach().numpy())

class nlp_utils():
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertLMHeadModel.from_pretrained('bert-base-cased')
        self.bert_model.eval()

        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_model.eval()


    def get_sequence_perplexity(self, s, model = 'bert',score_type = 'perplexity'):

        try:
            if model == 'bert':
                s = score(self.bert_tokenizer, self.bert_model, s, score_type)
            if model == 'gpt':
                s = score(self.gpt2_tokenizer, self.gpt2_model, s, score_type)
        except Exception as e:
            print("Error while get perplexity for: {} - {}".format(s,e))
            return np.nan
        return s
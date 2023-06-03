import numpy as np

from sentence_transformers import SentenceTransformer
import tensorflow as tf
#from senseBert.sensebert import SenseBert
import random
from configs.params import baseline_text_fn, synonyms_text_fn, vodka_text_fn
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead
from simpletransformers.language_representation import RepresentationModel
from simcse import SimCSE
import os, torch
from nlp_general.bert_finetuner import finetune_mlm
from configs.params import w2v_path
from gensim.models import KeyedVectors

w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)


class Embedder:
    def __init__(self, run_mlm_finetune):
        self.train_path = baseline_text_fn
        self.test_path = synonyms_text_fn
        self.eval_path = vodka_text_fn

        print("Take local sensebert")
        #self.sensebert_model = SenseBert("../materials/senseBert/sensebert-base-uncased", session=tf.compat.v1.Session())  # or sensebert-large-uncased
        # print("sensebert fetched")
        # sentence_transformers_stsb_bert_base_model = SentenceTransformer('stsb-bert-base')
        self.sentence_transformers_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
        # longformer_tokentizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        # longformer_model_base = AutoModel.from_pretrained("allenai/longformer-base-4096")
        self.longformer_fe_pipeline = pipeline('feature-extraction', model="allenai/longformer-base-4096", tokenizer="allenai/longformer-base-4096")
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')
        self.bert_representation_model = RepresentationModel(
            model_type="bert",
            model_name="bert-base-uncased",
            use_cuda=False
        )
        self.sim_sce_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        if run_mlm_finetune:
            self.bert_base_finetuned_model, self.bert_base_finetuned_tokenzier = finetune_mlm(cond="synonyms")
        else:
            self.bert_base_finetuned_model, self.bert_base_finetuned_tokenzier = None, None



    def get_sentence_embed(self,text,method):

        print("Get sentence embed of ", method)
        if method == 'random':
            enc = np.array([random.uniform(-1,2) for x in range(768)]).reshape(1,-1)

        if method == 'word2vec':
            l = []
            s = text.split(" ")
            for w in s:
                try:
                    l.append(w2v_model[w])
                except KeyError as e:
                    print("No w2v for " , w)
            enc = np.array(l).mean(axis=0).reshape(1,-1)

        if method == 'bert_sentence_representation':
            enc = self.bert_representation_model.encode_sentences([text], combine_strategy="mean").reshape(1,-1)

        if method == "bert_base_uncased":
            enc = get_pretrained_model_embed(self.bert_model, self.bert_tokenizer, text, lm_head=True)

        if method == "bert_base_uncased_finetuned":
            if self.bert_base_finetuned_model and self.bert_base_finetuned_tokenzier:
                enc = get_pretrained_model_embed(self.bert_base_finetuned_model, self.bert_base_finetuned_tokenzier,text , lm_head=True)
            else:
                enc = None

        if method == 'longformer':
            text_enc = self.longformer_fe_pipeline(text)
            sen_enc = text_enc[0][1:-1]
            enc = np.mean(sen_enc[1:-1], axis=0).reshape(1,-1)

        if method == "sentence-bert":
            enc = self.sentence_transformers_model.encode(text).reshape(1,-1)

        if method == 'sensebert':

            input_ids, input_mask = self.sensebert_model.tokenize([text])
            model_outputs = self.sensebert_model.run(input_ids, input_mask)
            contextualized_embeddings, mlm_logits, supersense_logits = model_outputs  # these are NumPy arrays
            enc = contextualized_embeddings[0][0].reshape(1,-1) #Take the first token ([CLS]) from each sentence

        if method == 'simCSE':
            enc = self.sim_sce_model.encode(text).numpy().reshape(1,-1)
        print(f"For {method} got enc with shape {enc.shape}")
        return enc

def get_pretrained_model_embed(model, tokenizer, input_sentences, lm_head):
    #https://medium.com/swlh/transformer-based-sentence-embeddings-cd0935b3b1e0

    def mean_pooling(model_output, attention_mask):
        # adapted from https://www.sbert.net/examples/applications/computing-embeddings/README.html
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[1][0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    encoded_input = tokenizer(input_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # compute token embeddings
    with torch.no_grad():
        # if lm_head:
        model_output = model(**encoded_input, output_hidden_states=True)#[1][0]
        # else:
        #     model_output = model(**encoded_input)

    # mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings.numpy()



from milky_nlp.nlp_preprocessing_utils import run_lemmatization, run_stemmization, clean_text, tokenize_text
from nlp_general.text_utils import split_textual_input_to_segments
import numpy as np
import joblib
from configs.params import ARTIFACTS_DIR
from nlp_general.embedding_utils import Embedder
from nlp_general.predictability_utils import get_sequence_perplexity
import os
import pandas as pd
# from mlm.scorers import MLMScorer, LMScorer
# from mlm.models import get_pretrained
# import mxnet as mx
def read_text(fn):
    f = open(fn, "r")
    return f.read()


def add_tr_col(df):
    df["TR_raw"] = pd.factorize(pd.cut(df.end_scaled, bins=np.arange(0, df.end_scaled.max() + 1.5, 1.5)))[0]
    df["TR_aligned"] = [t + 3 for t in df["TR_raw"]]
    return df
#todo clean code
class NLP_engine:
    def __init__(self, text_fn, synonyms_text_fn, baseline_words_csv_fn, synonyms_words_csv_fn,sentences_tr_csv_fn, lemmatize= True, stemmize = True):
        # baseline_transcript = read_text(text_fn)
        # synonyms_transcript = read_text(synonyms_text_fn)

        # self.baseline_words_df = add_tr_col(pd.read_csv(baseline_words_csv_fn))
        # self.synonyms_words_df = add_tr_col(pd.read_csv(synonyms_words_csv_fn))
        # self.baseline_words_df = pd.read_csv(baseline_words_csv_fn)
        # self.synonyms_words_df = pd.read_csv(synonyms_words_csv_fn)
        self.tr_df = pd.read_csv(sentences_tr_csv_fn)
        # baseline_sentences = baseline_transcript.split(".")
        # self.sentences_synonyms = split_input(synonyms_text.split(" "), "sentences" , suffix="milky_synonyms", use_existing=[False, False])
        # self.sentences = split_input(text.split(" "), "sentences" , suffix="milky", use_existing=[False, False])
        # self.baseline_tokenized_text = self.preprocess_text(baseline_transcript, clean=True, stem=False, lemma=False, tokenize=False)[0]
        # self.synonyms_tokenized_text = self.preprocess_text(synonyms_transcript, clean=True, stem=False, lemma=False, tokenize=False)[0]

    def create_sequences(self, use_existing = [True, True]):
        self.sentences_by_tr_df = split_textual_input_to_segments(self.tr_df, "sentences", suffix="milky_baseline", use_existing=use_existing)
        # self.synonyms_by_tr_df = split_textual_input_to_segments(self.tr_df, "sentences", suffix="milky_synonyms", use_existing=use_existing)
        # self.baseline_sequences = split_textual_input_to_segments(self.baseline_words_df, "segments", suffix="milky_baseline", use_existing=use_existing)
        # self.synonyms_sequences = split_textual_input_to_segments(self.synonyms_words_df, "segments", suffix="milky_synonyms", use_existing=use_existing)
        return self.sentences_by_tr_df

    def preprocess_text(self, segment, clean, tokenize, lemma, stem, keep_dot = False):
        processed_segments = []
        if clean:
            segment = clean_text(segment, keep_dot)
        if tokenize:
            a1 = tokenize_text(segment, method='re')
            a2 = tokenize_text(segment, method='space')
            a3 = tokenize_text(segment, method='keras')
            a4 = tokenize_text(segment, method='gensim')
            a5 = tokenize_text(segment, method='spacy')
            a6 = tokenize_text(segment, method='nltk')
            segment = a3

        if lemma:
            segment = run_lemmatization(segment)
        if stem:
            segment = run_stemmization(segment)

        processed_segments.append(segment)
        return processed_segments


def add_embeddings_to_sequences(seqs, prefix, run_mlm_finetune, save, use_existing = True):

    data_fn =  "{}/{}_per_seq_feats_dict.jbl".format(ARTIFACTS_DIR, prefix)
    print("data_fn is {} - path exits= {} , use existing= {}".format(data_fn, os.path.exists(data_fn) , use_existing  ))
    if use_existing and os.path.exists(data_fn):
        sequences_dicts = joblib.load(data_fn)
        nlp_df = pd.DataFrame(map(vars, sequences_dicts))
        nlp_df = nlp_df.join(pd.DataFrame([*nlp_df.pop('feats_dict')]))
        return sequences_dicts , nlp_df

    embedder = Embedder(run_mlm_finetune = run_mlm_finetune)
    sequences_dicts = []

    # ctxs = [mx.cpu()]
    # mlm_scoring_bert_model, mlm_scoring_bert_vocab, mlm_scoring_bert_tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
    # mlm_scoring_gpt_model, mlm_scoring_gpt_vocab, mlm_scoring_gpt_tokenizer = get_pretrained(ctxs, 'gpt2-117m-en-cased')
    # bert_scorer = MLMScorer(mlm_scoring_bert_model, mlm_scoring_bert_vocab, mlm_scoring_bert_tokenizer, ctxs)
    # gpt2_scorer = MLMScorer(mlm_scoring_gpt_model, mlm_scoring_gpt_vocab, mlm_scoring_gpt_tokenizer, ctxs)
    #print("Iterate over {} sequences (type {})".format(len(seqs), prefix))
    for sequecne_idx, sequence in enumerate(seqs):
        if sequecne_idx % 25 == 0:
            print("seq # {} - {}".format(sequecne_idx , sequence.baseline_text))
        sequence.feats_dict = {}
        t_d = {"vodka" : sequence.vodka_text, "synonyms" : sequence.synonyms_text, "baseline" : sequence.baseline_text}
        for text_type in ["vodka","synonyms", "baseline"]:

            text = t_d[text_type]
            sequence.feats_dict[text_type] = {}
            sequence.feats_dict[text_type]["text"] = text
            # sequence.feats_dict[text_type]["mlm_scoring_bert"] = bert_scorer.score_sentences([text])
            # sequence.feats_dict[text_type]["mlm_scoring_gpt"] =gpt2_scorer.score_sentences([text])
            # sequence.synonyms_sentence = synonyms_sequences[sequecne_idx]

            sequence.feats_dict[text_type]["word2vec"] = embedder.get_sentence_embed(text, method='word2vec')
           # sequence.feats_dict[text_type]["sensebert"] = embedder.get_sentence_embed(text, method='sensebert')
            sequence.feats_dict[text_type]["bert_base_finetuned"] = embedder.get_sentence_embed(text, method='bert_base_uncased_finetuned')
            sequence.feats_dict[text_type]["bert_base_nottuned"] = embedder.get_sentence_embed(text, method='bert_base_uncased')
            sequence.feats_dict[text_type]["simCSE"] = embedder.get_sentence_embed(text, method='simCSE')
            sequence.feats_dict[text_type]["longformer"] = embedder.get_sentence_embed(text, method='longformer')
            sequence.feats_dict[text_type]["bert"] = embedder.get_sentence_embed(text, method='bert_sentence_representation')
            sequence.feats_dict[text_type]["sentence-bert"] = embedder.get_sentence_embed(text, method='sentence-bert')
            sequence.feats_dict[text_type]["random"] = embedder.get_sentence_embed(text, method="random")
            # sequence.feats_dict[text_type]["perplexity"] = get_sequence_perplexity(text,"bert")
        sequences_dicts.append(sequence)

    print("Done!!")
    if save:
        joblib.dump(sequences_dicts, data_fn)
        print("Saved sequences_dicts to ", data_fn)
    nlp_df = pd.DataFrame(map(vars, sequences_dicts))
    nlp_df = nlp_df.join(pd.DataFrame([*nlp_df.pop('feats_dict')]))

    return sequences_dicts, nlp_df
            # sequence.feats_dict["w2v_embed_finetuned"] = get_w2v_embed(sequence.text)
            # sequence.feats_dict["bert_embed_finetuned"] = get_sentence_embed(sequence.text, method='bert' , finetune = True)

            ##############
            #
            # tokenized_segment = self.preprocess_text(segment_text)[0]
            # self.per_segment_dict[segment_idx] = {}
            # self.per_segment_dict[segment_idx]["orig_text"] = segment_text
            # self.per_segment_dict[segment_idx]["preprocessed_tokens"] = tokenized_segment
            # self.per_segment_dict[segment_idx]["sentiment"] = extract_sentiment(segment_text)
            #
            # self.per_segment_dict["words_dict"] = {}
            #
            # for word_idx, word in enumerate(tokenized_segment):
            #     wkey = (segment_idx, word_idx)
            #     self.per_segment_dict["words_dict"][wkey] = {}
            #     self.per_segment_dict["words_dict"][wkey]["word_pos"] = word_pos
            #
            #     self.per_segment_dict["words_dict"][wkey]["nback_5_words"] = self.tokenized_text[max(0, word_pos - 5) : word_pos ]
            #     self.per_segment_dict["words_dict"][wkey]["nback_10_words"] = self.tokenized_text[max(0, word_pos - 10) : word_pos ]
            #     self.per_segment_dict["words_dict"][wkey]["word_frequency"] = get_word_frequency(word, "freq")
            #     self.per_segment_dict["words_dict"][wkey]["word_zipf_frequency"] = get_word_frequency(word, "zipf")
            #     self.per_segment_dict["words_dict"][wkey]["bert_embedding"] = get_word_embed(word, "bert")
            #     self.per_segment_dict["words_dict"][wkey]["word_bert_pr_5nback"] = get_bert_word_in_sentence_prob(tokenized_segment, word, nback_5)
            #     self.per_segment_dict["words_dict"][wkey]["word_bert_pr_10back"] = get_bert_word_in_sentence_prob(tokenized_segment, word, nback_10)
            #     # self.per_word_dict[wkey]["word_elmo_pr"] = word_prob(word, "elmo")
            #     self.per_segment_dict["words_dict"][wkey]["word_bert_pr_rank"] = ""
            #     self.per_segment_dict["words_dict"][wkey]["elmo_bert_pr_rank"] = ""
            #
            #     # self.per_word_dict[wkey]["elmo_embedding"] = get_word_embed(word, "elmo")
            #     self.per_segment_dict["words_dict"][wkey]["w2v_embedding"] = get_word_embed(word, "w2v")
            #     self.per_segment_dict["words_dict"][wkey]["glove_embedding"] = get_word_embed(word, "glove")
            #     self.per_segment_dict["words_dict"][wkey]["bert_embedding_finetuned"] = word_embed(word, "elmo")
            #     self.per_segment_dict["words_dict"][wkey]["elmo_embedding_finetuned"] = word_embed(word, "elmo")
            #     self.per_segment_dict["words_dict"][wkey]["generated sentence"] = "" #we will generate next 3-5 words and see if they are activated
            #     self.per_segment_dict["words_dict"][wkey]["entity_recog"] = ""
            #     self.per_segment_dict["words_dict"][wkey]["wordnet_feats"] = ""
            #
            # self.per_segment_dict[segment_idx]["w2v_mean"] = ""
            # self.per_segment_dict[segment_idx]["bert_embedding"] = ""
            # self.per_segment_dict[segment_idx]["elmo_embedding"] = ""
            # self.per_segment_dict[segment_idx]["bert_embedding_finetuned_to_sentiment"] = ""
            # self.per_segment_dict[segment_idx]["elmo_embedding_finetuned_to_sentiment"] = ""
            # self.per_segment_dict[segment_idx]["GPT_perplexity"] = get_sentence_perplexity_gpt2(segment_text)
            # self.per_segment_dict[segment_idx]["berg_perplexity"] = get_sentence_perplexity_bert(segment_text)
            # self.per_segment_dict[segment_idx]["entity_recog"] = ""
            # self.per_segment_dict[segment_idx]["referenced_segment_idx"] = ""
            # if segment_idx > 0:
            #     self.per_segment_dict[segment_idx]["similarity_to_prev_sent"] = similarity_between_sents(segment_text,
            #                                                                                  self.segments[segment_idx - 1])
            # word_pos += 1
            # #todo add - visualize all sentences
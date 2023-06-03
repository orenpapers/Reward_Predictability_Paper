import sys
import numpy as np
# sys.path.insert(1, '/Users/orenkobo/Desktop/PhD/Aim2_new/HP_code/')
from nlp_general.sentiment_utils import  extract_sentiment
import pandas as pd
from configs.params import ARTIFACTS_DIR
import joblib

class Sequence:
    def __init__(self, idx, sequence_type, word_indices = None):
        self.sentence_idx = idx
        self.word_indices = word_indices
        self.sequence_type = sequence_type

    def has_word(self, word_idx):
        return word_idx in self.word_indices

    def get_feats_list(self):
        return self.pred_feats + self.sentiment_feats


def split_to_sentences(tr_sentences_df):
    segements = []
    for sentence_idx, seq_row in tr_sentences_df.iterrows():
        sequence = Sequence(sentence_idx, sequence_type="sentence")

        sequence.start_onset = seq_row.sec_from
        sequence.end_onset = seq_row.sec_to
        sequence.tr_start = seq_row.TR_from
        sequence.tr_end = seq_row.TR_to
        sequence.len_tr = seq_row.TR_len
        sequence.baseline_text = seq_row.baseline_sentence
        sequence.synonyms_text = seq_row.synonym_sentence
        sequence.vodka_text = seq_row.vodka_sentence
        # sequence.baseline_pred_feats = extract_sequence_predictability(sequence.segments_baseline_text) todo uncomment
        # sequence.baseline_sentiment_feats = extract_sentiment(sequence.segments_baseline_text)
        segements.append(sequence)
        if sentence_idx % 10 == 0:
            print("sentence_idx {} : {} ({} to {})".format(sequence.tr_start, sequence.baseline_text,
                                                           sequence.tr_start , sequence.tr_end))
            a = 2
    return segements

def split_to_segments(text_df):
    segement_idx = 0
    segements = []
    from nlp_general.predictability_utils import get_word_masked_predictability, extract_sequence_predictability, extract_word_predicatbility

    text_df = text_df.dropna(subset=["start_raw"]) #todo ask Yaara what to do with these words
    for tr, tr_words in text_df.groupby(["TR"]):
        if tr == -1:
            continue
        segement_words_indices = list(tr_words.index)
        segement_words = list(tr_words.word1.values)
        segement_words2 = list(tr_words.word2.values)
        segement_text = " ".join(segement_words)
        sequence = Sequence(segement_idx, segement_text, segement_words_indices, sequence_type="segment")
        sequence.pred_feats = extract_sequence_predictability(sequence.text)
        sequence.sentiment_feats = extract_sentiment(sequence.text)
        sequence.words = segement_words
        sequence.words2 = segement_words2
        sequence.TR = tr
        sequence.TR_aligned = tr_words.iloc[0].TR_aligned
        sequence.TR_raw = tr_words.iloc[0].TR_raw
        sequence.start_time = tr_words.iloc[0].start_scaled
        sequence.end_time = tr_words.iloc[0].end_scaled
        sequence.start_time_unscaled = tr_words.iloc[0].start_raw
        sequence.end_time_unscaled = tr_words.iloc[0].end_raw
        sequence.per_word_sentence_idx_mask = [] #todo if necessary - ask Aya
        segements.append(sequence)
        if tr % 10 == 0:
            print("TR {} : {} ({} to {})".format(tr, segement_text, sequence.start_time , sequence.end_time))
    print("TR {} : {} ({} to {})".format(tr, segement_text, sequence.start_time , sequence.end_time))
    return segements

def split_textual_input_to_segments(text_df, split_type, suffix, use_existing = [True, True]):

    fn = ARTIFACTS_DIR + "{}_{}_new.jbl".format(split_type, suffix)
    print("fn of segments is  ", fn)

    if split_type == "sentences":
        if use_existing[0]:
            s = joblib.load(fn)
            print("Loaded {} {} from {}".format(len(s), split_type, fn))
        else:
            s = split_to_sentences(text_df)
            joblib.dump(s, fn)
            print("Saved {} {} to {}".format(len(s), split_type, fn))

    if split_type == "segments":
        if use_existing[0]:
            s = joblib.load(fn)
            print("Loaded {} {} from {}".format(len(s), split_type, fn))
        else:
            s = split_to_segments(text_df)
            joblib.dump(s, fn)
            print("Saved {} {} to {}".format(len(s), split_type, fn))


            print("Saved {} {} to {}".format(len(s), split_type, fn))
    return s

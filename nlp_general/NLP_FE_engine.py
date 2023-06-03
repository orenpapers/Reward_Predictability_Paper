from nlp.predictability_utils import get_word_masked_predictability, extract_sequence_predictability, extract_word_predicatbility
from nlp.sentiment_utils import  extract_sentiment
import pandas as pd
# word_vectors_glove = api.load("glove-wiki-gigaword-100")
# word_vectors_w2v = api.load('word2vec-google-news-300')
# glove_sentemces_model = SentenceTransformer('average_word_embeddings_glove.840B.300d')
from configs.params import MATERIALS_DIR, SENTIMENT_BASE_FEATS, PREDICTABILITY_BASE_FEATS, TEXT_INPUT_TYPES
from nlp.text_utils import split_input
SENTIMENT_FEATS = [ '{}_{}'.format(x,y) for x in TEXT_INPUT_TYPES for y in SENTIMENT_BASE_FEATS]
PREDICTABILITY_FEATS = ['{}_{}'.format(x,y) for x in TEXT_INPUT_TYPES for y in PREDICTABILITY_BASE_FEATS]

oov_words = []

# class Sequence:
#     def __init__(self, idx, sequence_text, word_indices, sequence_type):
#         self.idx = idx
#         self.text_raw = sequence_text
#         self.text = self.text_raw.strip().replace("+","").replace("@","")
#         self.word_indices = word_indices
#         self.sequence_type = sequence_type
#
#     def has_word(self, word_idx):
#         return word_idx in self.word_indices
#
#     def get_feats_list(self):
#         return self.pred_feats + self.sentiment_feats
#
# def split_to_segments(words_input):
#     segement_idx = 0
#     segements = []
#     rows = []
#
#     for word_idx, word in enumerate(words_input):
#         if word_idx >= 5:
#             segement_text = " ".join(words_input[word_idx-4 : word_idx])
#             segement_idx += 1
#             segements.append(segement_text)
#             print("Sentence #{} : {}".format(segement_idx, segement_text))
#             segment_word_indices = words_input[word_idx-4 : word_idx]
#             sequence = Sequence(segement_idx, segement_text, segment_word_indices, sequence_type="segment")
#             sequence.pred_feats = extract_sequence_predictability(sequence.text)
#             sequence.sentiment_feats = extract_sentiment(sequence.text)
#             segements.append(sequence)
#             if segement_idx % 35 == 0:
#                 print("Added segment {} - {}".format(segement_idx, segement_text))
#         # row = [sequence, sentence_pred_feats, sentence_sentiment_feats]
#             # rows.append(row)
#     # segment_df = pd.DataFrame(columns = ["segment_idx","segment_text","word_indices_in_sentence"],
#     #                             data = rows)
#     # return segment_df
#     return segements
#
# def split_to_sentences(words_input):
#     sentence_idx = 0
#     sentence_start_indices = []
#     sentence_text = ""
#     sentences = []
#     rows = []
#     sentence_word_indices = []
#     for word_idx, word in enumerate(words_input):
#         if '.' in word:
#             sentence_idx += 1
#             sentence_start_indices.append(word_idx)
#             word = word[:-1]
#             sentence_text += ' ' + word
#             sentence_word_indices.append(word_idx)
#             sentence = Sequence(sentence_idx, sentence_text, sentence_word_indices, sequence_type="sentence")
#             sentence.pred_feats = extract_sequence_predictability(sentence.text)
#             sentence.sentiment_feats = extract_sentiment(sentence.text)
#             sentences.append(sentence)
#             if sentence_idx % 35 == 0:
#                 print("Added segment {} - {}".format(sentence_idx, sentence_text))
#             # row = [sequence, sentence_pred_feats, sentence_sentiment_feats]
#             # rows.append(row)
#             sentence_text = ""
#             # eos = True
#             sentence_word_indices = []
#         else:
#             sentence_text += ' ' + word
#             sentence_word_indices.append(word_idx)
#
#     # sentences_df = pd.DataFrame(columns = ["sentence_idx","sentence_text","word_indices_in_sentence"],
#     #                             data = rows)
#     return sentences
#
#
# def split_input(words_input, split_type, use_existing = [False, False]):
#
#     if split_type == "segments":
#         if use_existing[0]:
#             s = []
#         else:
#             s = split_to_segments(words_input)
#
#     if split_type == "sentences":
#         if use_existing[0]:
#             s = []
#         else:
#             s = split_to_sentences(words_input)
#
#     return s

class NLP_FeaturesExtractor:

    def __init__(self, words):
        self.words = words
        self.segments = split_input(words, "segments", suffix="HP")
        self.sentences = split_input(words, "sentences", suffix="HP")

        print("Got Total of {} segments and {} sentences")

    def get_feats_df(self):
        #https://github.com/EricFillion/happy-transformer/issues/196
        rows = []
        for word_idx, word in enumerate(self.words):

            word_pred_feats = extract_word_predicatbility(self.words, word_idx, word)
            word_sentiment_feats = extract_sentiment(word)
            segment = [x for x in self.segments if x.has_word(word_idx)]
            sentence = [x for x in self.sentences if x.has_word(word_idx)]
            row = [word, word_idx ] + word_pred_feats + word_sentiment_feats + [segment] + segment.get_feats_list() + \
                                                                               [sentence] + sentence.get_feats_list()
            rows.append(row)

        df = pd.DataFrame(data = rows,
                          columns=['word', 'word_idx', 'masked_pred_5back', 'masked_pred_10back','masked_pred_allback',
                                   'segment'] + SENTIMENT_FEATS + PREDICTABILITY_FEATS)
        return df












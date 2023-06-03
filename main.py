import joblib, datetime
from fmri.fmri_data_handler import FmriDataHandler
from configs.params import baseline_text_fn, synonyms_text_fn, fmri_data_path, synonyms_words_csv_fn, \
    baseline_words_csv_fn, baseline2synontms_withTR_fn, scrambled_vodka_fmri_data_path_new
from configs.params import synonyms_subject_ids, baseline_subject_ids, vodka_baseline_subject_ids, vodka_scrambled_subject_ids#, reward_anat_mask


# synonyms_subject_ids = ['130','131']
# baseline_subject_ids = ['104','081']
# vodka_subject_ids = ["023", "030"]
# anat_mask = reward_anat_mask
dataset_backgroud = ""
#alignment_dict_fn = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_standardize_detrend.jbl"#_withwholebrain2.jbl"
# qqq = joblib.load(alignment_dict_fn)
# a = 2
DEMO = False
if DEMO:
    synonyms_subject_ids = synonyms_subject_ids[:2]
    # baseline_subject_ids = baseline_subject_ids[:2] 
    vodka_baseline_subject_ids = vodka_baseline_subject_ids[:2]
    vodka_scrambled_subject_ids = vodka_scrambled_subject_ids[:2]
    baseline_subject_ids = baseline_subject_ids[:2]
    
# from configs.params import alignment_dict_fn_onlyaim2 as alignment_dict_fn
from configs.params import alignment_dict_fn_onlyaim2_with_ling_areas as alignment_dict_fn
print("Hello1111111")

def create_dict(use_existing_aligned_df = True, fetch_aim2 = True, fetch_aim3 = False, save = False,
                add_vodka_perplexity = True, use_nlp = True):
    from fmri.Fmri2NLP_DataCombiner import CombinedDataHandler

    # tr = EncodingModelTrainer(synonyms_subject_ids, train_iters = 10, model_params={}, demo=True)
    # ev = EncodingModelEvaluator(semantic_anat_mask = anat_mask, dataset_backgroud = dataset_backgroud)
    # ar = EncodingModelResultsAnalyser()
    
    if use_existing_aligned_df:

        alignment_dict = joblib.load(alignment_dict_fn)
        print("Loaded alignment_dict from ", alignment_dict_fn)
        return alignment_dict

    else:
        print(f"{datetime.datetime.now()} start - alignment_dict (will save to {alignment_dict_fn}")
        if use_nlp:
            print(f"{datetime.datetime.now()} start - nlp engine")
    

            from milky_nlp.nlp_features_extractor_manager import NLP_engine, add_embeddings_to_sequences
            
            ne = NLP_engine(baseline_text_fn, synonyms_text_fn, baseline_words_csv_fn, synonyms_words_csv_fn,
                            sentences_tr_csv_fn = baseline2synontms_withTR_fn)
            sentences = ne.create_sequences(use_existing=[True,True])
            sequences_enriched_dicts, nlp_df = add_embeddings_to_sequences(seqs = sentences, prefix = 'sentences_enriched',
                                                                           run_mlm_finetune = True, use_existing=True,
                                                                           save = True) #todo use simpleTransformers for encodings : https://simpletransformers.ai/docs/text-rep-examples/#minimal-example-for-generating-sentence-embeddings
            
            print(f"{datetime.datetime.now()} : Got NLP df with the shape of {nlp_df.shape}")
            if add_vodka_perplexity:
                from aim2_nlp_utils import nlp_utils
                print(f"{datetime.datetime.now()} : Add vodka perplexity")
                nu = nlp_utils()
                nlp_df["vodka_perplexity_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x))
                nlp_df["vodka_bert_entropy_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'bert','entropy'))
                nlp_df["vodka_bert_perplexity_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'bert','perplexity'))
                nlp_df["vodka_gpt_perplexity_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'gpt','perplexity'))
                nlp_df["vodka_gpt_entropy_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'gpt','entropy'))

            print(f"{datetime.datetime.now()} : Got shape of {nlp_df.shape}")
            a = 2
        else:
            nlp_df = None
        print(f"{datetime.datetime.now()} start - fmri data handler")
        fr = FmriDataHandler(fmri_data_path, scrambled_vodka_fmri_data_path_new, baseline_subject_ids, synonyms_subject_ids,
                             vodka_baseline_subject_ids, vodka_scrambled_subject_ids, nlp_df = nlp_df)

        if fetch_aim2:
            aim2_fmri_data_dict = fr.get_aim2_per_subject_fmri_data(use_existing = True, take_whole_brain = False,
                                                                    take_ling_semantic = True, take_ling_comprehenstion = True,
                                                                    take_reward = True,
                                                                    take_audio_control = False, take_vision_control = True, save=False)
        else:
            aim2_fmri_data_dict = {}
        if fetch_aim3:
            aim3_fmri_data_dict , tr2sentences_dict, sen2tr_dict = fr.get_aim3_per_subject_fmri_data(use_existing = True) #take only
        else:
            aim3_fmri_data_dict , tr2sentences_dict, sen2tr_dict = {} , {}, {}
        # fmri_data = fr.process_signal(fmri_data ,apply_detrend = True, apply_add_HRF = False, apply_reduce_mean = False)

        print(f"{datetime.datetime.now()} start - create alignment dict")
        dh = CombinedDataHandler()
        alignment_dict = dh.align_fmri_nlp(nlp_df , aim3_fmri_data_dict, aim2_fmri_data_dict, tr2sentences_dict, sen2tr_dict,
                                           # fr.whole_brain_mask, fr.fitted_reward_brain_mask,
                                      #     fr.fitted_vision_mask, fr.fitted_hearing_mask,
                                           use_existing=True)
        print(f"{datetime.datetime.now()}  - created alignment dict")
        if save:
            print(datetime.datetime.now(), " Saving")
            joblib.dump(alignment_dict,alignment_dict_fn )
            print(f"{datetime.datetime.now()} Dumped alignment_dict to {alignment_dict_fn}")
            exit()
        #    exit()
        return alignment_dict
    # exit()
    # results_dict = tr.train_model_on_df(alignment_dict, use_existing_results_dict = False, use_existing_models = False, roi='all')
    # print("results_dict is " , results_dict)

def main():
    create_dict()

if __name__ == '__main__':
    main()


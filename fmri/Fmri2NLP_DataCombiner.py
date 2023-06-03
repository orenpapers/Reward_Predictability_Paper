import pandas as pd
from configs.params import ARTIFACTS_DIR
import joblib, os

class CombinedDataHandler:

    def __init__(self):
        pass

    def align_fmri_nlp(self, nlp_df , aim3_fmri_data_dict , aim2_fmri_data_dict, tr2sentences_dict, sen2tr_dict, 
                       # fitted_whole_brain_mask, fitted_reward_brain_mask,
                       # fitted_vision_mask, fitted_hearing_mask,
                       use_existing):

        alignment_dict_fn = ARTIFACTS_DIR + "fmri_nlp_alignment_allsubjs_withwholebrain.jbl"
        if use_existing and os.path.exists(alignment_dict_fn):
            alignment_dict = joblib.load(alignment_dict_fn)
            return alignment_dict

        alignment_dict = {}
        alignment_dict['tr2sentences_dict'] = tr2sentences_dict
        alignment_dict['sen2tr_dict'] = sen2tr_dict
        alignment_dict['aim3_fmri_data_dict'] = aim3_fmri_data_dict
        alignment_dict['aim2_fmri_data_dict'] = aim2_fmri_data_dict
        alignment_dict['nlp_data_df'] = nlp_df
        # alignment_dict['fitted_wholebrain_mask'] = fitted_whole_brain_mask
        # alignment_dict['fitted_reward_mask'] = fitted_reward_brain_mask
        # alignment_dict["fitted_hearing_mask"] = fitted_hearing_mask
        # alignment_dict["fitted_vision_mask"] = fitted_vision_mask
        return alignment_dict

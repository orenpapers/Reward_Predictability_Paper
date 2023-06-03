import datetime

from nilearn.input_data import  NiftiMasker
import nilearn
from nilearn.datasets import load_mni152_brain_mask
from configs.params import ARTIFACTS_DIR, comprehenstion_mask, semantic_mask, reward_mask
import joblib
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.image import clean_img

from configs.params import ctrl_audio_mask, ctrl_vision_mask

class FmriDataHandler:

    def __init__(self, fmri_data_path, scrambled_vodka_fmri_data_path, baseline_subject_ids, synonyms_subject_ids, vodka_baseline_subject_ids,
                 vodka_scrambled_subject_ids, nlp_df, dataset_mask_path =""):
        self.fmri_data_path = fmri_data_path
        self.baseline_subject_ids = baseline_subject_ids
        self.scrambled_vodka_fmri_data_path = scrambled_vodka_fmri_data_path
        self.synonyms_subject_ids = synonyms_subject_ids
        self.vodka_intact_subject_ids = vodka_baseline_subject_ids
        self.vodka_scrambled_subject_ids = vodka_scrambled_subject_ids

        audio_img = nib.load(ctrl_audio_mask)
        self.audio_img = image.math_img("img>0", img=audio_img)

        reward_img = nib.load(reward_mask)
        self.reward_img = image.math_img("img>0", img=reward_img)

        vision_img = nib.load(ctrl_vision_mask)
        self.vision_img = image.math_img("img>0", img=vision_img)

        self.mni_152_dataset_img = load_mni152_brain_mask()

        comprehenstion_img = nib.load(comprehenstion_mask)
        self.comprehenstion_img = image.math_img("img>0", img=comprehenstion_img)

        semantic_img = nib.load(semantic_mask)
        self.semantic_img = image.math_img("img>0", img=semantic_img)

        a = 2
        self.nlp_df = nlp_df
        if nlp_df is not None:
            self.nlp_df['aligned_TRs_of_sentence'] = self.nlp_df.apply(lambda x: list(range(x.tr_start -1 , x.tr_end)), axis=1)
            self.tr2sentences_dict = pd.Series(self.nlp_df["aligned_TRs_of_sentence"],
                                               index=self.nlp_df["sentence_idx"].values).to_dict()
            a = 2

    def get_aim2_atlas_data(self):
        from nilearn.datasets import fetch_atlas_harvard_oxford
        ho_atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
        a = 2



    def get_aim2_per_subject_fmri_data(self, use_existing, take_whole_brain = True, take_reward = True,
                                       take_ling_semantic = True, take_ling_comprehenstion = True,
                                       take_audio_control = False, take_vision_control = True, save = False):
        d = {}
        fn_suf = "aim2_fmri_data_dict_FINAL_nodetrend_withLing"
        if take_whole_brain:
            fn_suf += "_wholebrain_NEWFMRIPREP"
        else:
            fn_suf += "_no_wholebrain_NEWFMRIPREP"
        d_fn = ARTIFACTS_DIR + fn_suf + ".jbl"
        print("Take existing from " , d_fn)
        import os
        print("get_aim2_per_subject_fmri_data use existing = ", use_existing)
        if use_existing and os.path.exists(d_fn) :
            d = joblib.load(d_fn)
            print("loaded AIM2 per subjects fmri data dict from ", d_fn)
            return d

        d['vodka'] = {}
        d['vodka']["vodka-intact-reward"] = {}
        d['vodka']["vodka-intact-audio"] = {}
        d['vodka']["vodka-intact-vision"] = {}
        d['vodka']["vodka-intact-wholebrain"] = {}
        d['vodka']['vodka-intact-comprehenstion'] = {}
        d['vodka']['vodka-intact-semantic'] = {}
        d['vodka']["vodka-scrambled-reward"] = {}
        d['vodka']["vodka-scrambled-audio"] = {}
        d['vodka']["vodka-scrambled-vision"] = {}
        d['vodka']["vodka-scrambled-wholebrain"] = {}
        d['vodka']['vodka-scrambled-comprehenstion'] = {}
        d['vodka']['vodka-scrambled-semantic'] = {}
        d['vodka']['fitted_brain_masks'] = {}

        print("===== READING INTACT =====")
        for si, subj_id in enumerate(self.vodka_intact_subject_ids):
            # d['vodka_baseline'][subj_id] = {}
            func_dir = self.fmri_data_path

            print(f"{datetime.datetime.now()} Read INTACT AIM2 fmri data of subject {subj_id} from {func_dir} ({si}/{len(self.vodka_scrambled_subject_ids)})")

            # subj_func_file = f'{func_dir}/sub-{subj_id}/func/sub-{subj_id}_task-milkyway_desc-preproc_bold.nii.gz'
            subj_func_file = f'{func_dir}/sub-{subj_id}/func/sub-{subj_id}_task-milkyway_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz'
            regressor_file = f'{func_dir}/sub-{subj_id}/func/sub-{subj_id}_task-milkyway_desc-confounds_regressors.tsv'
            print(f"{datetime.datetime.now()}: Read {subj_func_file}")
            if take_whole_brain:
                task_type = "vodka-intact-wholebrain"
                subj_fmri_baseline_data_df, c = self.get_data(subj_id, subj_func_file, mask_img=self.mni_152_dataset_img,
                                                           regressor_file = regressor_file)
                print(f"{datetime.datetime.now()} Intact, {task_type }  - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")
                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_audio_control:

                task_type = "vodka-intact-audio"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.audio_img,
                                                           regressor_file = regressor_file)
                print(f"{datetime.datetime.now()} Intact, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")
                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_vision_control:
                task_type = "vodka-intact-vision"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.vision_img,
                                                           regressor_file = regressor_file)
                print(f"{datetime.datetime.now()} Intact, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")
                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c


            if take_reward:
                task_type = "vodka-intact-reward"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.reward_img,
                                                           regressor_file = regressor_file)
                print(f"{datetime.datetime.now()} Intact, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")
                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_ling_comprehenstion:
                task_type = "vodka-intact-comprehenstion"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.comprehenstion_img,
                                                               regressor_file = regressor_file)

                print(f"{datetime.datetime.now()} Take ling, Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")

                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_ling_semantic:
                task_type = "vodka-intact-semantic"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.semantic_img,
                                                               regressor_file = regressor_file)

                print(f"{datetime.datetime.now()} Take ling, Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")

                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

        print("===== READING SCRAMBLED =====")
        for si, subj_id in enumerate(self.vodka_scrambled_subject_ids):
            func_dir = self.scrambled_vodka_fmri_data_path

            print(f"{datetime.datetime.now()} Read SCRAMBLED AIM2 fmri data of subject {subj_id} from {func_dir} ({si}/{len(self.vodka_scrambled_subject_ids)})")
            # print(f"{datetime.datetime.now()} Read AIM2 fmri data of subject {subj_id} from {self.vodka_fmri_data_path}")
            # subj_func_file = f'{func_dir}/sub-{subj_id}/func/sub-{subj_id}_task-vodkascram_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
            subj_func_file = f'{func_dir}/sub-{subj_id}/func/sub-{subj_id}_task-vodkascram_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz'
            regressor_file = f'{func_dir}/sub-{subj_id}/func/sub-{subj_id}_task-vodkascram_desc-confounds_regressors.tsv'

            if take_whole_brain:
                task_type = "vodka-scrambled-wholebrain"
                subj_fmri_baseline_data_df, c = self.get_data(subj_id, subj_func_file, mask_img=self.mni_152_dataset_img,
                                                           regressor_file = regressor_file)
                print(f"{datetime.datetime.now()} Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")
                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c


            if take_audio_control:
                task_type = "vodka-scrambled-audio"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.audio_img,
                                                           regressor_file = regressor_file)
                print(f"{datetime.datetime.now()} Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")
                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_vision_control:
                task_type = "vodka-scrambled-vision"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.vision_img,
                                                           regressor_file = regressor_file)
                print(f"{datetime.datetime.now()} Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")
                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_reward:
                task_type = "vodka-scrambled-reward"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.reward_img,
                                                           regressor_file = regressor_file)

                print(f"{datetime.datetime.now()} Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")

                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_ling_comprehenstion:
                task_type = "vodka-scrambled-comprehenstion"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.comprehenstion_img,
                                                               regressor_file = regressor_file)

                print(f"{datetime.datetime.now()} Take ling, Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")

                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

            if take_ling_semantic:
                task_type = "vodka-scrambled-semantic"
                subj_fmri_baseline_data_df, c  = self.get_data(subj_id, subj_func_file, mask_img=self.semantic_img,
                                                               regressor_file = regressor_file)

                print(f"{datetime.datetime.now()} Take ling, Scramble, {task_type } - Reshaped {task_type} to {subj_fmri_baseline_data_df.shape}")

                d['vodka'][task_type][subj_id] = subj_fmri_baseline_data_df
                if task_type not in d['vodka']['fitted_brain_masks']:
                    d['vodka']['fitted_brain_masks'][task_type] = c

        from configs.params import MASKS_DIR
        if save:
            print(f"{datetime.datetime.now()} Saving AIM2 per subjects fmri data dict to {d_fn}")
            joblib.dump(d, d_fn)
            print(f"{datetime.datetime.now()} Saved AIM2 per subjects fmri data dict to {d_fn}")
            for cond_key, fitted_brain_mask in d['vodka']['fitted_brain_masks'].items():
                joblib.dump(d['vodka']['fitted_brain_masks']['vodka-intact-reward'], f"{MASKS_DIR}/fitted_mask_{cond_key}_MEW.jbl")
                print(f"Dumped fitted brain mask of {cond_key} to {MASKS_DIR}")
            # joblib.dump(d['vodka']['fitted_brain_masks']['vodka-intact-reward'], MASKS_DIR + "fitted_mask_vodka_intact_reward.jbl")
            # joblib.dump(d['vodka']['fitted_brain_masks']['vodka-scrambled-reward'], MASKS_DIR + "fitted_mask_vodka_scrambled_reward.jbl")
            # joblib.dump(d['vodka']['fitted_brain_masks']['vodka-intact-vision'], MASKS_DIR + "fitted_mask_vodka_intact_vision.jbl")
            # joblib.dump(d['vodka']['fitted_brain_masks']['vodka-scrambled-vision'], MASKS_DIR + "fitted_mask_vodka_scrambled_vision.jbl")
        return d

    def get_aim3_per_subject_fmri_data(self, nii_desc ="desc-preproc_bold", use_existing = True):
        d = {}
        #todo for nii_desc, use sub-001_task-MGT_run-01_bold_space-MNI152NLin2009cAsym_preproc.nii.gz #see here - https://lukas-snoek.com/NI-edu/fMRI-introduction/week_7/nilearn_stats.html
        from aim3_analysis.analyse_new_aim3 import transform_tr_df_to_sentence_df

        d_fn = ARTIFACTS_DIR + "subjects_fmri_data_dict_allsubjs.jbl"
        tr2sentences_dict_fn = ARTIFACTS_DIR + "tr2sents_dict.jbl"
        sentence2tr_dict_fn = ARTIFACTS_DIR + "sent2tr_dict.jbl"
        dataset_mask_fn = ARTIFACTS_DIR + "dataset_mask.jbl"

        if use_existing:
            d = joblib.load(d_fn)
            print("loaded per subjects fmri data dict from ", d_fn)
            tr2sentences_dict = joblib.load(tr2sentences_dict_fn)
            sen2tr_dict = joblib.load(sentence2tr_dict_fn)
            # dataset_mask = joblib.load(dataset_mask_fn)
            return d, tr2sentences_dict , sen2tr_dict

        # 14 TR : music + gray screen
        # 5 TR : HRF moving
        #https://nilearn.github.io/auto_examples/02_decoding/plot_miyawaki_encoding.html#sphx-glr-auto-examples-02-decoding-plot-miyawaki-encoding-py
        sids = self.vodka_intact_subject_ids + self.baseline_subject_ids + self.synonyms_subject_ids
        print(f"{len(sids)} Subjects are : {sids}")
        d['vodka'] = {}
        d['milkyway'] = {}
        d['synonyms'] = {}
        for s_i, subj_id in enumerate(sids):
            # d[subj_id] = {}
            if subj_id in self.vodka_intact_subject_ids:
                task_type = "vodka"
            if subj_id in self.baseline_subject_ids:
                task_type = "milkyway"
            if subj_id in self.synonyms_subject_ids:
                task_type = "synonyms"

            print("{} Read (aim3) fmri data of subject #{} {} ({}) from {} ".format(datetime.datetime.now(), s_i, subj_id, task_type, self.fmri_data_path))

            subj_func_dir = '{}/sub-{}/'.format(self.fmri_data_path,subj_id)
            subj_func_file = f'{subj_func_dir}/func/sub-{subj_id}_task-milkyway_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz'
            regressor_file = f'{subj_func_dir}/func/sub-{subj_id}_task-milkyway_desc-confounds_regressors.tsv'
            subj_fmri_baseline_data_df_semantic, _ = self.get_data(subj_id, subj_func_file, mask_img=self.semantic_img,
                                                       regressor_file = regressor_file)
            subj_fmri_baseline_data_df_comp, _ = self.get_data(subj_id, subj_func_file, mask_img=self.comprehenstion_img,
                                                                   regressor_file = regressor_file)

            subj_fmri_baseline_data_df_viz, _ = self.get_data(subj_id, subj_func_file, mask_img=self.vision_img,
                                                               regressor_file = regressor_file)

            subj_fmri_baseline_data_df_reward, _ = self.get_data(subj_id, subj_func_file, mask_img=self.reward_img,
                                                              regressor_file = regressor_file)

            d[task_type][subj_id] = {}
            d[task_type][subj_id]["semantic_fmri_TR_df"] = subj_fmri_baseline_data_df_semantic
            d[task_type][subj_id]["comprehention_fmri_TR_df"] = subj_fmri_baseline_data_df_comp
            d[task_type][subj_id]["vision_fmri_TR_df"] = subj_fmri_baseline_data_df_viz
            d[task_type][subj_id]["reward_fmri_TR_df"] = subj_fmri_baseline_data_df_reward

            d[task_type][subj_id]["semantic_fmri_sentence_df"] = transform_tr_df_to_sentence_df(subj_fmri_baseline_data_df_semantic, self.tr2sentences_dict,
                                                                                     method='mean')
            d[task_type][subj_id]["comprehention_fmri_sentence_df"] = transform_tr_df_to_sentence_df(subj_fmri_baseline_data_df_comp, self.tr2sentences_dict,
                                                                                          method='mean')
            d[task_type][subj_id]["vision_fmri_sentence_df"] = transform_tr_df_to_sentence_df(subj_fmri_baseline_data_df_viz, self.tr2sentences_dict,
                                                                                                     method='mean')
            d[task_type][subj_id]["reward_fmri_sentence_df"] = transform_tr_df_to_sentence_df(subj_fmri_baseline_data_df_reward, self.tr2sentences_dict,
                                                                                                     method='mean')
            # d[task_type]["task_type"] = task_type
            a = 2

        sen2tr_dict = {}
        for tr, sens in self.tr2sentences_dict.items():
            for sen in sens:
                sen2tr_dict[sen] = tr

        joblib.dump(d, d_fn)
        joblib.dump(self.tr2sentences_dict, tr2sentences_dict_fn)
        joblib.dump(sen2tr_dict, sentence2tr_dict_fn)
        # joblib.dump(dataset_mask , dataset_mask_fn)
        print("Saved AIM3 per subjects fmri data dict to ", d_fn)
        return d, self.tr2sentences_dict , sen2tr_dict

    def get_data(self, subj_id, nii_file, mask_img, regressor_file = None):

        if regressor_file:
            confounds_df = pd.read_csv(regressor_file, delimiter='\t', index_col=False)
            confounds_cols = ['framewise_displacement']

            if pd.isna(confounds_df['framewise_displacement'].iloc[0]):
                confounds_df['framewise_displacement'].iloc[0] = 0

            confounds_arr = confounds_df[confounds_cols].values
        else:
            confounds_arr = None

        c = NiftiMasker(mask_img = mask_img, smoothing_fwhm=6)
        w = c.fit_transform(nii_file)

        if subj_id in ["105","123","038"]:
            w = w[range(10,279),:]
            confounds_arr = confounds_arr[range(10,279),:]
            # subj_fmri_data_df = subj_fmri_data_df.iloc[10:279].reset_index(drop=True)
        else:
            w = w[range(14,283),:]
            confounds_arr = confounds_arr[range(14,283),:]
            # subj_fmri_data_df = subj_fmri_data_df.iloc[14:283].reset_index(drop=True)

        d = nilearn.signal.clean(w,  standardize='zscore', t_r = 1.5, detrend=True, high_pass=0.007,confounds=confounds_arr)
        subj_fmri_data_df = pd.DataFrame(d)
        #     a = 2
        #
        #
        #     c1 = NiftiMasker(mask_img = mask_img, detrend=False, standardize=True, high_pass=0.007, t_r = 1.5, smoothing_fwhm=6)
        #     w1 = c1.fit_transform(nii_file, confounds=confounds_df[confounds_cols].values)
        #
        #     subj_fmri_data_df1 = pd.DataFrame(w1)
        #     if subj_id in ["105","123","038"]:
        #         subj_fmri_data_df1 = subj_fmri_data_df1.iloc[10:279].reset_index(drop=True)
        #     else:
        #         subj_fmri_data_df1 = subj_fmri_data_df1.iloc[14:283].reset_index(drop=True)
        #     a = 2
        # except Exception as e:
        #     a = 1
        # clean_bold_nii = clean_img(nii_file, detrend=True, standardize=True, confounds=confounds_arr,
        #                            high_pass=0.007, t_r = 1.5  )
        # print("standardize=True")
        # print(f"{datetime.datetime.now()}: {subj_id}: Got pre-fitted bold with the shape {clean_bold_nii.shape}")
        # subj_fmri_data = self.fitted_audio_mask.transform(clean_bold_nii)

        # print(f"{datetime.datetime.now()} Got fitted bold with shape {subj_fmri_data_df.shape}")
        return subj_fmri_data_df, c

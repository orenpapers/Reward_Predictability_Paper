 
from uuid import getnode as get_mac

fmri_data_path   = "/Users/orenkobo/Desktop/PhD/Aim3/narratives/data/narratives/derivatives/fmriprep/"
scrambled_vodka_fmri_data_path = r"C:\Users\orenk\Desktop\Aim2_Resources/Materials/vodka/deriviatives/fmriprep/"
scrambled_vodka_fmri_data_path_new = r"C:\Users\orenk\Desktop\Aim2_Resources/Materials/vodka/deriviatives_new/deriviatives/fmriprep"
ARTIFACTS_DIR = r"C:/Users/orenk/Desktop/Aim2_Resources/Artifacts/"
MATERIALS_DIR = r"C:\Users\orenk\Desktop\Aim2_Resources/materials/"
MASKS_DIR =  r"C:\Users\orenk\Desktop\Aim2_Resources/materials/Anatomic_Mask/"
aim2_code_artifacts_dir = "/Users/orenkobo/Desktop/PhD/Aim3/narratives/Aim2_code_artifacts/"
aim2_analysis_artifacts_dir = ARTIFACTS_DIR
# aim3_analysis_artifacts_dir = "/Users/orenkobo/Desktop/PhD/Aim3/Analysis_artifacts/13032022/"

EVALUATION_ARTIFACTS_DIR = ARTIFACTS_DIR + "Evaluation_artifacts/"
baseline_text_fn = MATERIALS_DIR + "transcripts/milkywayoriginal_transcript.txt"
synonyms_text_fn = MATERIALS_DIR + "transcripts/milkywaysynonyms_transcript.txt"
vodka_text_fn    = MATERIALS_DIR + "transcripts/milkywayvodka_transcript.txt"
synonyms_words_csv_fn = MATERIALS_DIR + "baseline_align_enriched_withTR.csv"
baseline_words_csv_fn = MATERIALS_DIR + "synonyms_align_enriched_withTR.csv"
baseline2synontms_withTR_fn = MATERIALS_DIR + "baseline2synonyms2vodka_sentence_withTR.csv"
# alignment_dict_fn = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_FINAL_withaim3_newfmriprep_newproc.jbl"#_withwholebrain2.jbl"
# alignment_dict_fn_onlyaim2 = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_140322_onlyAim2.jbl"
alignment_dict_fn_onlyaim2 = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_120422_onlyAim2.jbl"
alignment_dict_fn_onlyaim2_with_ling_areas = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_120422_onlyAim2_with_ling_areas.jbl"
alignment_dict_fn_onlyaim_wb = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_120422_only_wb.jbl"
# alignment_dict_fn_onlyaim3 = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_140322_onlyAim3.jbl" #Had some good results with this
alignment_dict_fn_onlyaim3 = ARTIFACTS_DIR + "fmri2nlp_alignment_dict_230322_onlyAim3.jbl" #Added viz and reward as ctrl
wb_dict = "/Users/imac/Desktop/PhD/Aim3/narratives/artifacts/aim2_fmri_data_dict_FINAL_nodetrend_wholebrain_NEWFMRIPREP.jbl"
synonyms_subject_ids = ['115', '116', '117', '118', '119', '120', '121', '122', '131',
                        '123', '124', '125', '126', '127', '128', '129', '130']

baseline_subject_ids = ['034', '081', '083', '091', '093', '095', '096', '099', '100',
                        '101', '104', '105', '106', '108', '111', '112', '113', '114']

vodka_baseline_subject_ids = ["023", "030", "032", "038", "052", "079", "086", "087", "088",
                              "089", "090", "097", "098", "102", "103", "107", "109", "110"]

vodka_scrambled_subject_ids = ["088", "115", "116", "117", "118", "119", "120", "121", "122",
                            "123", "124", "125", "126", "127", "128", "129", "130", "131"]

ENC_MODELS  = ['simCSE','longformer',"bert", "random",
               "sentence-bert"]#,"sensebert"] #"word2vec","bert_base_finetuned", 'bert_base_nottuned',

w2v_path = r"C:\Users\orenk\Desktop\Aim2_Resources/word2vec/GoogleNews-vectors-negative300.bin.gz"


bg_brain = MASKS_DIR + 'bg.nii.gz'
# reward_anat_mask = "/Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/Reward_masks/nifti_roi-masks_fig09/binConjunc_PvNxDECxRECxMONxPRI_striatum.nii.gz"
reward_mask = MASKS_DIR + "reward_association-test_z_FDR_0.01_neurosynth.nii.gz"
# mni_mask = MASKS_DIR + "MNI152_T1_3mm_brain_mask.nii"

comprehenstion_mask = MASKS_DIR + "Semantic_masks/comprehension_association-test_z_FDR_0.01.nii.gz"
semantic_mask = MASKS_DIR + "Semantic_masks/semantic_association-test_z_FDR_0.01.nii.gz"
# 13448  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig03/fig03A/I_POS_kda10_testStat_mean_cMass_testStat_bin.nii.gz
# 5339  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig03/fig03B/I_NEG_kda10_testStat_mean_cMass_testStat_bin.nii.gz
# 2944  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig03/fig03C/conjunc_POSxNEG_minTestStat_bin.nii.gz
# 4297  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig03/fig03D/C_POSvsNEG_kda10_clust_tstat1_bin.nii.gz
# 1104  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig06/fig06A/I_DECISION_POS_striatum.nii.gz
# 2046  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig06/fig06A/I_DECISION_POS_vmpfc.nii.gz
# 500  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig09/binConjunc_PvNxDECxRECxMONxPRI_striatum.nii.gz
# 449  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/nifti_roi-masks_fig09/binConjunc_PvNxDECxRECxMONxPRI_vmpfc.nii.gz
# 3339  :  /Users/orenkobo/Desktop/PhD/Aim3/narratives/materials/Anatomic_Mask/derivatives_model_model001_masks_SVC_vmpfc_mask.nii.GZ
# hearing_mask = MASKS_DIR + "hearing_association-test_z_FDR_0.01.nii.gz"
# vision_mask = MASKS_DIR + "vision_association-test_z_FDR_0.01.nii.gz"
ctrl_vision_mask = MASKS_DIR + "primary_visual_association-test_z_FDR_0.01.nii.gz"
ctrl_audio_mask = MASKS_DIR + "auditory_association-test_z_FDR_0.01.nii.gz"


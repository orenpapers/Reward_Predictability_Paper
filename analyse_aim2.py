import datetime
import random

import scipy.stats
from aim2_nlp_utils import nlp_utils
from sklearn.metrics import mean_squared_error
from configs.params import ARTIFACTS_DIR
import nibabel
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import time
from sklearn.metrics import accuracy_score
from sklearn.decomposition import SparsePCA, FastICA, LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import LeavePOut
from configs.params import alignment_dict_fn_onlyaim3
from sklearn.linear_model import LinearRegression
from configs.params import synonyms_subject_ids, vodka_baseline_subject_ids, baseline_subject_ids
import joblib
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from mne.stats import fdr_correction
from mne.stats import fdr_correction
from mne.stats import bonferroni_correction
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.spatial import distance
# from nlp_general.predictability_utils import extract_sequence_predictability
from nilearn.plotting import plot_stat_map
from nilearn.image import coord_transform
from nilearn.image import threshold_img
from nilearn.connectome import ConnectivityMeasure
import configs.params
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, wilcoxon, shapiro
import string
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import GridSearchCV
from configs import params
from scipy import stats
from scipy.stats import norm
from nilearn.glm import threshold_stats_img
from configs.params import aim2_code_artifacts_dir, aim2_analysis_artifacts_dir
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import median_absolute_error, explained_variance_score, r2_score

ALL_ESTIMATORS = [
  #  KNeighborsClassifier(n_neighbors=6), KNeighborsClassifier(n_neighbors=12), KNeighborsClassifier(n_neighbors=30),SVC(kernel='rbf',probability=True)
    SVC(kernel='linear',probability=True),  #SVC(kernel='poly',probability=True),
    RandomForestClassifier(n_estimators=100), #RandomForestClassifier(n_estimators=200),
    # LogisticRegression(solver = 'liblinear', penalty='l1', max_iter=1000), LogisticRegression(penalty='l2',max_iter=1000),
    # LogisticRegression(C=0.2 , max_iter=1000), LogisticRegression(C=0.8 , max_iter=1000)
]

ALL_FSS = [  PCA(20),  PCA(10)#, PCA(50),#, PCA(128),
            #SelectKBest(f_classif, k=10), SelectKBest(f_classif, k=15),
             # FastICA(n_components=10), FastICA(n_components=20)
             ]


# DATA_DIR = "/Users/orenkobo/Desktop/PhD/Aim2_new/Materials/vodka/deriviatives/"

def get_mean_activation_per_voxel(alignment_dict):

    d = alignment_dict["aim2_fmri_data_dict"]
    baseline_dfs = [d['vodka']['vodka-intact-reward'][x] for x in params.vodka_baseline_subject_ids]
    vodka_dfs = [d['vodka']['vodka-scrambled-reward'][x] for x in params.vodka_scrambled_subject_ids]
    baseline_mean_df = pd.concat(baseline_dfs).mean(level=0)
    vodka_reward_dfs = pd.concat(vodka_dfs).mean(level=0)
    return baseline_mean_df, vodka_reward_dfs

def get_per_voxel_diff_between_conditions(baseline_mean_df, vodka_mean_df, out_fn, save = True ):
    per_voxel_diff = []
    voxels = list(baseline_mean_df.columns)
    for voxel in voxels:
        baseline_mean_voxel_bold = baseline_mean_df.iloc[:, voxel]
        vodka_mean_voxel_bold = vodka_mean_df.iloc[:, voxel]
        assert(len(baseline_mean_voxel_bold) == 269)
        assert(len(vodka_mean_voxel_bold) == 269)
        activities_distance = distance.euclidean(baseline_mean_voxel_bold , vodka_mean_voxel_bold) #todo use mahalanobis for distances
        per_voxel_diff.append(activities_distance)
    if save:
        joblib.dump(per_voxel_diff, out_fn)
        print("Saved get_per_voxel_diff_between_conditions to ", out_fn)
    return per_voxel_diff

def generate_null_distribution_distances(baseline_mean_df, vodka_mean_df, num_iters,
                                         null_dist_fn,
                                         save = True):
    #Fig 1b here : https://journals.sagepub.com/doi/pdf/10.1177/0956797616682029
    print(datetime.datetime.now(), "generate_null_distribution_distances")

    null_distribution_distances = []
    unified_df = pd.concat([baseline_mean_df, vodka_mean_df])
    assert (len(baseline_mean_df) == len(vodka_mean_df) == 269)
    for i in range(num_iters):
        if (i % 10) == 0:
            print(f"{datetime.datetime.now()} : Generating null distribution, iter# {i}/{num_iters} - shape of df to shuffle is {unified_df.shape} ")
        randomized_unified_df = shuffle(unified_df).reset_index(drop=True)
        randomized_baseline = randomized_unified_df[:len(baseline_mean_df)]
        randomized_vodka = randomized_unified_df[len(baseline_mean_df):]
        per_voxel_randomized_dist = get_per_voxel_diff_between_conditions(randomized_baseline, randomized_vodka, out_fn= None, save = False)
        null_distribution_distances.append(per_voxel_randomized_dist)

    print(f"{datetime.datetime.now()} : Done")
    null_distribution_distances_df = pd.DataFrame(data = null_distribution_distances)
    print(f"{datetime.datetime.now()} - create null_distribution_distances_df with shape {null_distribution_distances_df.shape}")
    if save:
        print(f"{datetime.datetime.now()} saving null distribution")
        null_distribution_distances_df.to_csv(null_dist_fn, index=False)
        print(f"{datetime.datetime.now()} Saved csv to {null_dist_fn}")
        try:
            null_distribution_distances_df.to_feather(null_dist_fn + ".fthr")
        except Exception as e:

            print(f"Cant write feature - {e} , retry")
            null_distribution_distances_df.columns = [str(x) for x in null_distribution_distances_df.columns]
            null_distribution_distances_df.to_feather(null_dist_fn + ".fthr")
            print(f"{datetime.datetime.now()} Saved fthr to {null_dist_fn}")
        print(f"{datetime.datetime.now()} null distribution saved to {null_dist_fn}")
    return null_distribution_distances_df

def calc_p_val(null_distribution_distances, dist):
    return null_distribution_distances.index(dist)

def get_per_voxels_significance_level(null_distribution_distances, per_voxel_distance_list, mc_method):
    #https://stackoverflow.com/questions/69388717/pandas-how-to-check-percentile-of-each-element-of-a-line-at-a-corresponding-colu
    print(f"{datetime.datetime.now()} - get_per_voxels_significance_level")
    per_voxel_rank = null_distribution_distances.lt(per_voxel_distance_list).sum().tolist()
    print(f"{datetime.datetime.now()} - Got list with len {len(per_voxel_rank)}")
    per_voxel_rank = [x+1 for x in per_voxel_rank]
    per_voxel_raw_p = [x/len(null_distribution_distances) for x in per_voxel_rank]

    d = pd.DataFrame([per_voxel_distance_list] , columns = null_distribution_distances.columns)
    print(f"{datetime.datetime.now()} - Created d with shape {d.shape}")
    per_voxel_z = pd.concat([d, null_distribution_distances]).apply(lambda x : stats.zscore(x) , axis=0).iloc[0] #this takes ~10mins
    print(f"{datetime.datetime.now()} - Got per voxel z")
    per_voxel_cdf = null_distribution_distances.apply(lambda x : stats.norm(x.mean(), x.std()) , axis=0) #this takes ~10mins
    per_voxel_p = [x[0].cdf(x[1]) for x in zip(per_voxel_cdf, per_voxel_distance_list)]
    print(f"{datetime.datetime.now()} - Got per voxel p")

    if mc_method == "fdr_i":
        per_voxel_fdr_corrected_is_significance, per_voxel_adjusted_p = fdr_correction(per_voxel_p, method='indep')
    if mc_method == "fdr_c":
        per_voxel_fdr_corrected_is_significance, per_voxel_adjusted_p = fdr_correction(per_voxel_p, method='negcorr')
    if mc_method == "bonferroni":
        per_voxel_fdr_corrected_is_significance, per_voxel_adjusted_p = bonferroni_correction(per_voxel_p)
    print(f"{datetime.datetime.now()} - Finished significance tests")
    return per_voxel_adjusted_p, per_voxel_raw_p, per_voxel_z

def analyse_connectivity(alignment_dict, per_voxel_fdr_corrected_p_val, fig_fn, rand = False):
    from sklearn.linear_model import LogisticRegression
    #https://nilearn.github.io/auto_examples/03_connectivity/plot_group_level_connectivity.html#what-kind-of-connectivity-is-most-powerful-for-classification
    print(f"{datetime.datetime.now()} : Analyse connectivity")
    d = alignment_dict["aim2_fmri_data_dict"]
    subj_vec = [x for x in d.values()]
    subjects_data = [x["masked_fmri_data"].values for x in subj_vec]
    subjects_data = [x[:, per_voxel_fdr_corrected_p_val] for x in subjects_data]
    task_types = [x['task_type'] for x in subj_vec]
    kinds = ['correlation', 'partial correlation','covariance','tangent','precision']
    print(f"data is a list of {len(subjects_data)} elements with shapes {[x.shape for x in list(subjects_data)]}")
    # cv = StratifiedShuffleSplit(n_splits=50, random_state=0, test_size=8)

    scores = []
    rand_scores = []
    X = np.array(subjects_data)
    y = np.array(task_types)
    for kind in kinds:
        connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
        Xc = connectivity.fit_transform(X)
        for i in range(100):
            X_train, X_test, y_train, y_test = train_test_split(Xc,y, test_size=0.2, random_state=42*i)

            print(f"{datetime.datetime.now()} : Got measures")
            classifier = LogisticRegression().fit(X_train, y_train)
            print(f"{datetime.datetime.now()} : Traind classifier")

            score = classifier.score(X_test, y_test) #accuracy_score(y_test, y_pred)
            y_rand = y_test.copy()
            random.shuffle(y_rand)
            rand_score = classifier.score(X_test, y_rand)
            rand_scores.append(rand_score)
            scores.append(score)

            print(f"{datetime.datetime.now()} : Acc is {score}, rand acc is {rand_score}")
            plt.figure(figsize=(4, 3))

        plt.boxplot([scores, rand_scores])
        plt.xticks([1,2], ['Actual', 'dummy'])
        plt.title('Prediction: accuracy score')
        plt.savefig(fig_fn + "_" + kind + ".png")
        print("Saved per-subject connectivity classifier to ", fig_fn)

def brain_plot(masker, per_voxel_alpha, bg_brain, fig_fn, st_name):
    # bring the scores into the shape of the background brain

    score_map_img = masker.inverse_transform(per_voxel_alpha)
    thresholded_score_map_img = threshold_img(score_map_img, threshold=0, copy=False)
    if st_name == 'zscore':
        score_map_img_inverse = masker.inverse_transform([-x for x in per_voxel_alpha])
    if st_name == 'adjusted_p':
        score_map_img_inverse = masker.inverse_transform([1-x for x in per_voxel_alpha])
    thresholded_score_map_img_inverse = threshold_img(score_map_img_inverse, threshold=0, copy=False)



    return score_map_img, score_map_img_inverse#, thresholded_score_map_img, thresholded_score_map_img_inverse

def cluster_thershold_plot(zscores, fig_dir, masker, c, h, n):

    zscores_mask = masker.inverse_transform(zscores)

    thresholded_map1, threshold1 = threshold_stats_img(zscores_mask,
                                         #              threshold=norm.isf(0.001),
                                                       cluster_threshold=10, two_sided=False
    )

    plot_stat_map(thresholded_map1,  threshold=threshold1, output_file=f"{fig_dir}{c}_{h}_{n}_cluster_thr(10).png",
                  display_mode='z',  black_bg=True,
                  title=f'{c} - {h} corrected,  Z threshold (Cluster size = 10))')


def trdf2sendf(tr_df, sen2tr_dict):
    return tr_df.groupby(sen2tr_dict).mean()

def train_cross_subject_perplexity_classifier(alignment_dict,demo,conds, output_dir):
    output_dir += "cross_subject_perplexity_classifier/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv = 2 if demo else 5
    d = alignment_dict["aim2_fmri_data_dict"]['vodka']
    nlp_df = alignment_dict["nlp_data_df"]
    sen2tr_dict = alignment_dict["sen2tr_dict"]
    vodka_perplexity_median = nlp_df['vodka_perplexity_score'].median()
    res_rows = []

    for c in conds:
        print(f"{datetime.datetime.now()} : {c} : train_cross_subject_classifier - stacking per-subject data")
        if c == 'perm':
            task_0_key = f'vodka-intact-reward'
        else:
            task_0_key = f'vodka-intact-{c}'
        # task_ctrl_key = f'vodka-intact-{ctrl_task}'
        intact_subjs = d[task_0_key].keys()
        X = pd.DataFrame()
        y = []
        for i, subj in enumerate(intact_subjs):
            # print(f"{datetime.datetime.now()} : subj {subj} ({c} - train_cross_subject_classifier ({i}/{len(intact_subjs)})")
            subj_data = d[task_0_key][subj]
            subj_X = trdf2sendf(subj_data, sen2tr_dict)
            subj_y = nlp_df['vodka_perplexity_score'].ge(vodka_perplexity_median).tolist()

            X = pd.concat([X , subj_X])
            y += subj_y


        conf = 0
        print(f"{datetime.datetime.now()} : {c} -  Data Collected, training  ...")
        for fs in [SelectKBest(f_classif, k=10) , PCA(10)]:
            for est in [SVC(kernel='linear'), LogisticRegression()]:
                print(f"{datetime.datetime.now()} : {c} : conf #{conf} - {fs}, {est}")
                svc_ovo = make_pipeline(fs, est)

                if c == "perm":
                    print("A")
                    cv_scores_ovo = permutation_test_score(est, X, y, cv=cv, n_permutations = 3)[1]
                    print("B")
                else:
                    cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=ShuffleSplit(n_splits=50, test_size=0.1, random_state=0), verbose=1)

                acc = round(np.mean(cv_scores_ovo),2)
                st = round(np.std(cv_scores_ovo), 2)
                print(f"{datetime.datetime.now()} -{c} ({cv} folds), {subj} ({fs} , {est})- Got {acc} (+-{st}) ({cv_scores_ovo})")
                res_rows.append([conf, fs, est, acc, st, c, cv_scores_ovo.tolist()])
                plt.figure(figsize=(4, 3))
                plt.boxplot([cv_scores_ovo])
                plt.axhline(0.5)
                plt.xticks([1], ['Actual'])
                plt.title('accuracy score')
                plt.savefig(f"{output_dir}/{c}_conf#{conf}.png")
                plt.close()
                conf += 1


    res_df = pd.DataFrame(data = res_rows, columns = ["conf","fs","est","acc","std", "c", "cv_scores"])
    best_conf = res_df.sort_values(['acc']).conf.iloc[-1]
    best_conf_res_df = res_df[res_df.conf == best_conf]
    cv_res_df = best_conf_res_df.explode("cv_scores")
    print(f"{datetime.datetime.now()} : Generating perm")
    perm_scores = permutation_test_score(est, X, y, cv=3, n_permutations = 1000)[1].tolist()
    joblib.dump(perm_scores , f"{output_dir}/perm_scores_cross_subject_perplexity_classifier.jbl")
    print(f"{datetime.datetime.now()} : Generated perm")
    print(res_df)
    p_val = (len([x for x in perm_scores if best_conf_res_df.iloc[0].acc < x]) -1) / len(perm_scores)
    print("REWARD PVAL IN train_cross_subject_perplexity_classifier is ", p_val)
    for conf_i, conf_df in cv_res_df.groupby(["conf"]):
        ax = sns.swarmplot(x="conf",hue='c', y="cv_scores",data=conf_df, palette="Set2")
        ax.figure.savefig(f"{output_dir}/best_conf(#{best_conf})_acc_score_of_cross_subject_perplexity_classifier_swarm.png")
        ax.axhline(y=0.5)
        plt.title(f"reward p_val = {p_val}")
        plt.clf()
        plt.cla()
        ax = sns.boxplot(x="conf",hue='c', y="cv_scores",data=conf_df, palette="Set2")
        ax.figure.savefig(f"{output_dir}/best_conf(#{best_conf})_acc_score_of_cross_subject_perplexity_classifier_box.png")
        ax.axhline(y=0.5)
        plt.title(f"reward p_val = {p_val}")
        plt.clf()
        plt.cla()
    print("Saved figs to ", output_dir)
    return res_df


def train_multi_sentence_classifier(alignment_dict, demo, conds, output_dir):

    d = alignment_dict["aim2_fmri_data_dict"]['vodka']
    nlp_df = alignment_dict["nlp_data_df"]
    sen2tr_dict = alignment_dict["sen2tr_dict"]
    res_rows = []
    i = 0
    for c in ['intact','scrambled']:
        it_key = f'vodka-{c}-reward'
        # task_ctrl_key = f'vodka-intact-{ctrl_task}'
        intact_subjs = d[it_key].keys()
        for i, loo_subj in enumerate(intact_subjs):
            train_subjects = [x for x in intact_subjs if x != loo_subj]
            X_data = [trdf2sendf(d[it_key][x], sen2tr_dict) for x in train_subjects]
            X_train = pd.concat(X_data).mean(level=0)
            y_train = [str(x) for x in range(len(X_train))]

            X_test = trdf2sendf(d[it_key][loo_subj], sen2tr_dict)
            y_test = [str(x) for x in range(len(X_test))]
            for fs in ALL_FSS:#[PCA(32), PCA(48) , PCA(15), PCA(20), SelectKBest(f_classif, k=10),  FastICA(n_components=6), FastICA(n_components=10), FastICA(n_components=15),FastICA(n_components=20)]:
            # for fs in [SelectKBest(f_classif, k=10) , PCA(10)]:
                for est in ALL_ESTIMATORS:#[SVC(kernel='linear'), LogisticRegression(max_iter=1000)]:
                    try:
                        print(f"{datetime.datetime.now()} -{c}: subject {loo_subj} (#{i}) run {fs} , {est}: {i}/ {2*18*len(ALL_FSS)*len(ALL_ESTIMATORS)}")

                        pipe = make_pipeline(fs, est)
                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_test)
                        acc = accuracy_score(y_pred, y_test)
                        row1 = [np.nan,i, c, loo_subj, "actual", fs, est, acc, list(y_pred), list(y_test)]
                        i+=1
                        for q in range(55):
                            y_train_shuffled = sorted(y_train, key=lambda k: random.random())
                            pipe = make_pipeline(fs, est)
                            pipe.fit(X_train, y_train_shuffled)
                            y_pred = pipe.predict(X_test)
                            acc = accuracy_score(y_pred, y_test)
                            row2 = [q,i, c, loo_subj, "shuffled", fs, est, acc, list(y_pred), list(y_test)]
                            res_rows.append(row2)

                        res_rows.append(row1)
                    except Exception as e:
                        print("Failed - ", e)

    res_df = pd.DataFrame(data = res_rows, columns=["shuffle_num","iter_num","cond","test_subj","train_type","fs","est","accuracy","y_pred","y_test"])

    return res_df


def train_within_subject_perplexity_classifier(alignment_dict,demo,conds, output_dir):
    output_dir += "within_subject_perplexity_classifier/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv = 2 if demo else 5
    d = alignment_dict["aim2_fmri_data_dict"]['vodka']
    nlp_df = alignment_dict["nlp_data_df"]


    nu = nlp_utils()
    # nlp_df["vodka_bert_entropy_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'bert','entropy'))
    nlp_df["vodka_bert_perplexity_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'bert','perplexity'))
    nlp_df["vodka_gpt_perplexity_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'gpt','perplexity'))
    # nlp_df["vodka_gpt_entropy_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'gpt','entropy'))
    sen2tr_dict = alignment_dict["sen2tr_dict"]

    res_rows = []
    iter = 0
    total_iters = 2 * 2 * 18 * len(ALL_FSS) * len(ALL_ESTIMATORS)
    y_cols = ['vodka_bert_perplexity_score']#,'vodka_gpt_perplexity_score']
    for y_col in y_cols:
        predictability_median = nlp_df[y_col].median()

        for cond in ['intact','scrambled']:
            for roi in ['reward','vision']:
                print(f"{datetime.datetime.now()} : {roi} : train_within_subject_perplexity_classifier - {y_col}")

                task_0_key = f'vodka-{cond}-{roi}'
                # task_ctrl_key = f'vodka-intact-{ctrl_task}'
                cond_subjs = d[task_0_key].keys()

                for i, subj in enumerate(cond_subjs):
                    print(f"{datetime.datetime.now()} : subj {subj} ({i}/{len(cond_subjs)})")
                    subj_data = d[task_0_key][subj]
                    X = trdf2sendf(subj_data, sen2tr_dict)
                    y = nlp_df[y_col].ge(predictability_median) #X here is 66, 12031
                    conf = 0
                    for fs in ALL_FSS:
                        for est in ALL_ESTIMATORS:
                            print(f"{datetime.datetime.now()} -conf#{conf}/{len(ALL_ESTIMATORS)*len(ALL_FSS)}: {roi}, {subj} ({fs} , {est}) (iter#{iter}/{total_iters}")

                            svc_ovo = make_pipeline(fs, est)
                            cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=ShuffleSplit(n_splits=500, test_size=0.2, random_state=0))
                            # null_cv_scores = permutation_test_score(svc_ovo, X, y, cv=5, n_permutations =20, random_state=0)[1]
                            res_rows.append([iter, subj, y_col, "actual", conf, fs, est, cond, roi, round(np.mean(cv_scores_ovo),2), round(np.std(cv_scores_ovo), 2),  cv_scores_ovo.tolist()])
                            # res_rows.append([iter, subj, y_col, "perm", conf, fs, est, cond, roi, round(np.mean(null_cv_scores),2), round(np.std(null_cv_scores), 2),  null_cv_scores.tolist()])
                            conf += 1
                            iter+=1

    res_df = pd.DataFrame(data = res_rows, columns = ["iter","subj_id","y_col","model_type","conf","fs","est","cond", "roi","acc", "std", "cv_scores"])
    print("Saved figs to ", output_dir)

    return res_df

def train_subject_cond_classifier(alignment_dict,demo,conds, level, out_dir, ctrl_task = 'audio'):
    cv = 2 if demo else 5
    #https://nilearn.github.io/auto_examples/03_connectivity/plot_group_level_connectivity.html#what-kind-of-connectivity-is-most-powerful-for-classification
    print(f"{datetime.datetime.now()} train_subject_cond_classifier, level is {level}")

    d = alignment_dict["aim2_fmri_data_dict"]['vodka']

    res_rows = []

    for c in conds:
        for dim in ['temporal']:#:,'spatial']:
            print("****", dim, "****")
            if dim == 'spatial':
                agg_axis = 0

            if dim == 'temporal':
                agg_axis = 1
            task_0_key = f'vodka-intact-{c}'
            task_1_key = f'vodka-scrambled-{c}'
            task_ctrl_key = f'vodka-intact-{ctrl_task}'

            intact_subjs = d[task_0_key].keys()
            scrmbled_subjs = d[task_1_key].keys()
            ctrl_subjs = d[task_ctrl_key].keys()

            subjects_data_0 = [d[task_0_key][x].values for x in intact_subjs]
            subjects_label_0 = [0] * len(intact_subjs)

            subjects_data_1 = [d[task_1_key][x].values for x in scrmbled_subjs]
            subjects_label_1 = [1] * len(scrmbled_subjs)

            subjects_data_ctrl = [d[task_ctrl_key][x].values for x in ctrl_subjs]
            subjects_label_ctrl = [1] * len(ctrl_subjs)

            if level == 'control':
                X0 = np.array([x.mean(agg_axis) for x in subjects_data_0])
                X1 = np.array([x.mean(agg_axis) for x in subjects_data_ctrl])

                y = np.array(subjects_label_0+subjects_label_ctrl)
                X = np.concatenate([X0, X1] , axis=0)

            if level == 'subject':
                X0 = np.array([x.mean(agg_axis) for x in subjects_data_0])
                X1 = np.array([x.mean(agg_axis) for x in subjects_data_1])

                y = np.array(subjects_label_0+subjects_label_1)
                X = np.concatenate([X0, X1] , axis=0)

            print(f"{datetime.datetime.now()} : shape of X is {X.shape}, y is {y.shape} ({dim}")
            #Shape of x is 36 (n_subjects) * 12031 (n_voxels) for spatial and 36X269 for temporal
            k = int(X.shape[1] / 20)
            print(f"{datetime.datetime.now()} : {c} - Take only {k} features ({dim})")

            conf = 0
            from sklearn.ensemble import RandomForestClassifier
            ests = [RandomForestClassifier(n_estimators=50), RandomForestClassifier(n_estimators=100), RandomForestClassifier(n_estimators=200),
                    SVC(kernel='linear', probability=True), SVC(kernel='rbf', probability=True), SVC(kernel='poly', probability=True),
                    LogisticRegression(solver = 'liblinear', penalty='l1', max_iter=1000), LogisticRegression(penalty='l2',max_iter=1000),
                    LogisticRegression(C=0.2 , max_iter=1000), LogisticRegression(C=0.8 , max_iter=1000)]
            for fs in [SelectKBest(f_classif, k=k) , PCA(10), PCA(15), PCA(20), PCA(24)]:#,
                         #FastICA(n_components=15), FastICA(n_components=10), FastICA(n_components=20) , FastICA(n_components=24)]:
                for est in ests:#[SVC(kernel='linear',random_state=0), LogisticRegression(max_iter=1000,random_state=0)]:
                    print(f"{datetime.datetime.now()} - run {fs} , {est}")
                    svc_ovo = make_pipeline(fs, est)

                    print(f"{datetime.datetime.now()} - run cross_val_score ({c}, {dim})")
                    # >>> n_samples = X.shape[0]
                    cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=ShuffleSplit(n_splits=1000, test_size=0.33, random_state=0))
                    null_cv_scores = permutation_test_score(est, X, y, cv=cv, n_permutations = 1000, random_state=0)[1]
                    joblib.dump(null_cv_scores , f"{out_dir}/{dim}_{c}_conf#{conf}subject_cond_classifier_permutation_scores_{cv}folds_250perms.jbl")

                    pval = (len([x for x in null_cv_scores if x >= np.mean(cv_scores_ovo)]) -1) / len(null_cv_scores)
                    # cv_scores_ovo2 = cross_val_score(svc_ovo, X, y, cv=cv, verbose=1)
                    print(f"{datetime.datetime.now()} -{c} - Got {np.mean(cv_scores_ovo)} ) {dim}")
                    res_rows.append(["actual",dim, conf, fs, est, np.mean(cv_scores_ovo), c,pval, cv_scores_ovo.tolist()])
                    res_rows.append(["dummy",dim, conf, fs, est, np.mean(null_cv_scores), c, np.nan, null_cv_scores.tolist()])
                    conf+=1
    res_df = pd.DataFrame(data = res_rows, columns = ["model_type","agg_dim","conf","fs","est","res", "c", "pval","per_fold_score"])
    best_conf = res_df.sort_values(['res']).conf.iloc[-1]
    best_conf_res_df = res_df[(res_df.conf == best_conf) & (res_df.model_type=="actual")]
    cv_res_df = best_conf_res_df.explode("per_fold_score")
    return res_df

def train_encoding_model(alignment_dict,demo,conds, output_dir):

    alignment_dict_onlyaim3 = joblib.load(alignment_dict_fn_onlyaim3)
    output_dir += "encoding_model_classifier/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iters = 18 if demo else 1000
    d = alignment_dict["aim2_fmri_data_dict"]['vodka']
    nlp_feats_df = alignment_dict_onlyaim3['nlp_data_df']
    sen2tr_dict = alignment_dict["sen2tr_dict"]
    res_rows = []
    from configs.params import ENC_MODELS
    import math
    enc_models = ['longformer',"bert", "random",
                  "sentence-bert",]
    enc_models = ENC_MODELS
    estimators = [Ridge(alpha=1, random_state=7), Ridge(alpha=5, random_state=7),
                  # Ridge(alpha=10, random_state=7),
                  # Ridge(alpha=25, random_state=7), Ridge(alpha=50, random_state=7),
                  #Ridge(alpha=100, random_state=7)
                  ]
    rois = ['reward','vision']
    conds = ['intact','scrambled']
    conf = 0

    for enc_model in enc_models:
        for reg_estimator in estimators:
            for roi in rois:
                for cond in conds:
                    conf += 1
                    print(f"{datetime.datetime.now()} : {cond} : train_encoding_model (conf {conf}/{math.prod([len(x) for x in [enc_models,estimators,rois,conds]])}) ({enc_model}, {reg_estimator}, {roi}, {cond})")

                    task_0_key = f'vodka-{cond}-{roi}'

                    X = pd.DataFrame(np.vstack([d.get(enc_model) for d in list(nlp_feats_df.baseline)]))# pd.DataFrame(np.array([senc_model.encode(x) for x in nlp_df.vodka_text]))
                    cond_fmri_dict = {k:v for k,v in alignment_dict_onlyaim3['aim3_fmri_data_dict']['vodka'].items() if k in vodka_baseline_subject_ids}
                    y = pd.concat([x[f'{roi}_fmri_sentence_df'] for x in cond_fmri_dict.values()]).mean(level=0)#  pd.concat(X_subj_data_fmri).mean(level=0)
                    if cond == 'scrambled':
                        y = y.sample(frac=1).reset_index(drop=True)

                    lpo = LeavePOut(2)
                    n = lpo.get_n_splits(X)
                    for i, (train_index, test_index) in enumerate(lpo.split(X)):
                        if i == iters:
                            break
                        if i % 215 == 0:
                            print(f"{datetime.datetime.now()}: iter {i}(/{iters}) ({reg_estimator}, {cond}, {roi}: conf {conf})")
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        reg_estimator.fit(X_train, y_train)
                        y_pred = reg_estimator.predict(X_test)
                        r2_score_eval = r2_score(y_pred, y_test)
                        sen1_dist = distance.euclidean(y_test.iloc[0], y_pred[0,:])
                        sen1_dist_wrong = distance.euclidean(y_test.iloc[0], y_pred[1,:])
                        sen2_dist = distance.euclidean(y_test.iloc[1], y_pred[1,:])
                        sen2_dist_wrong = distance.euclidean(y_test.iloc[1], y_pred[0,:])
                        sen1_dist_eval = True if sen1_dist < sen1_dist_wrong else False
                        sen2_dist_eval = True if sen2_dist < sen2_dist_wrong else False
                        row = [roi, cond, enc_model, reg_estimator, train_index, test_index, r2_score_eval, sen1_dist, sen1_dist_wrong, sen1_dist_eval,
                               sen2_dist, sen2_dist_wrong, sen2_dist_eval, X_train.shape[1]]
                        res_rows.append(row)

    res_df = pd.DataFrame(res_rows, columns = ["roi","cond","encoding_model","regression_model", "train_idx","test_idx","r2score_eval",
                                                 "sen1_dist","sen1_dist_wrong","sen1_dist_eval",
                                                 "sen2_dist","sen2_dist_wrong","sen2_dist_eval", "nVoxels"])
    res_df.to_csv(output_dir + "/encoding_model_res.csv")
    return res_df

def analyse_brain(output_dir, c, alignment_dict_fn, num_iters = 100, save = True,
                  null_distributions_path = "/Users/imac/Desktop/PhD/Aim2_new/Analysis_artifacts/new_proc_FINAL/null_distributions",
                  use_existing = False):

    print(f"{datetime.datetime.now()}: analyse_brain")
    s = ""
    start_ts = str(int(time.time())) + "_"
    sc_mean_df_fn = f"{null_distributions_path}/{c}_NEW_sc_mean_df_{num_iters}_.csv"
    it_mean_df_fn = f"{null_distributions_path}/{c}_NEW_it_mean_df_{num_iters}_.csv"
    null_dist_fn =  f"{null_distributions_path}/null_distribution_distances_df_{c}_{num_iters}_shuffleX.fthr.fthr"
    per_voxel_actual_distance_list_fn= f"{null_distributions_path}/{c}_{num_iters}_actual_diffs_between_conds_l.jbl"
    masker_fn =  f"{null_distributions_path}/wb_masker_fitted_{c}.jbl"
    if use_existing:

        print(datetime.datetime.now(), "Take null dist from " , null_dist_fn)
        null_distribution_distances_df = pd.read_feather(null_dist_fn)
        print(datetime.datetime.now(), "Took shape " , null_distribution_distances_df.shape)
        per_voxel_actual_distance_list = joblib.load(per_voxel_actual_distance_list_fn)
        print(datetime.datetime.now(), "Got list size " , len(per_voxel_actual_distance_list))
        masker = joblib.load(masker_fn)
        print(datetime.datetime.now(), "Got Mask")
    else:
        print(f"{datetime.datetime.now()} : Create it, sc data")

        print(datetime.datetime.now(), "Load alignment_dict_fn_onlyaim_wb")
        alignment_dict = joblib.load(alignment_dict_fn)
        print(datetime.datetime.now())
        joblib.dump(alignment_dict['aim2_fmri_data_dict']['vodka']['fitted_brain_masks'][f'vodka-intact-{c}'], masker_fn)
        print(datetime.datetime.now())
        masker = joblib.load(masker_fn)

        print(datetime.datetime.now(), "Loaded alignment_dict_fn_onlyaim_wb_2")

        brain_data = alignment_dict['aim2_fmri_data_dict']['vodka']
        sc_data = [brain_data[f'vodka-scrambled-{c}'][s] for s in brain_data[f'vodka-scrambled-{c}'].keys()]
        it_data = [brain_data[f'vodka-intact-{c}'][s] for s in brain_data[f'vodka-intact-{c}'].keys()]
        print(f"{datetime.datetime.now()} : Created lists")
        sc_mean_df = pd.concat(sc_data).mean(level=0)
        it_mean_df = pd.concat(it_data).mean(level=0)
        print(f"{datetime.datetime.now()} : Created dfs with shapes {it_mean_df.shape}, {sc_mean_df.shape}")
        if save:
            print(f"{datetime.datetime.now()} : Saveing per cond mean df to {output_dir}")
            sc_mean_df.to_csv(sc_mean_df_fn, index=False)
            it_mean_df.to_csv(it_mean_df_fn, index=False)

        per_voxel_actual_distance_list = get_per_voxel_diff_between_conditions(baseline_mean_df= it_mean_df,
                                                                               vodka_mean_df= sc_mean_df,
                                                                               save = save,
                                                                               out_fn=per_voxel_actual_distance_list_fn)

        null_distribution_distances_df = generate_null_distribution_distances(baseline_mean_df = it_mean_df,
                                                                           vodka_mean_df= sc_mean_df,
                                                                           num_iters = num_iters,
                                                                           save = save,
                                                                           null_dist_fn = null_dist_fn,
                                                                           )


    for mc_method in ["fdr_i", "fdr_c", "bonferroni"]:
        print(f"{datetime.datetime.now()} =========== Run with {mc_method} ===========")
        per_voxel_adjusted_p, per_voxel_raw_p, per_voxel_z = get_per_voxels_significance_level(null_distribution_distances_df,
                                                                                           per_voxel_actual_distance_list,
                                                                                           mc_method)

        for st_per_voxel, st_name in zip([per_voxel_adjusted_p, per_voxel_z], ['adjusted_p','zscore']):
            print(f"{datetime.datetime.now()} - Making grpahs for ", st_name)
            joblib.dump(list(st_per_voxel), f"{output_dir}/{start_ts}_{c}_{mc_method}_{st_name}_list.jbl")
            print(f"{datetime.datetime.now() } Threshold plot to {output_dir}{c}")

            score_map_img, score_map_img_inverse =  brain_plot(per_voxel_alpha = st_per_voxel, masker = masker, st_name = st_name,
                       bg_brain = configs.params.bg_brain, fig_fn = output_dir + f"{c}_{mc_method}_per_voxel_{st_name}.png")

            thresholded_score_map_img = threshold_img(score_map_img, threshold=3.1, copy=False)
            thresholded_score_map_img_inverse = threshold_img(score_map_img_inverse, threshold=3.1, copy=False)
            print(f"{datetime.datetime.now() } Brain plot to {output_dir}{c}")


            nibabel.save(score_map_img, output_dir + start_ts + "_" + c + "_" +mc_method +"_" + st_name + ".nii")
            nibabel.save(score_map_img_inverse, output_dir + start_ts + "_" + c+  "_" + mc_method +"_" + st_name +"_inverse.nii")
            nibabel.save(thresholded_score_map_img, output_dir + start_ts + "_" + c + "_" +mc_method +"_" + st_name + "_threshold31.nii")
            nibabel.save(thresholded_score_map_img_inverse, output_dir + start_ts + "_" + c+  "_" + mc_method +"_" + st_name +"_inverse_threshold31.nii")

        s += f"{c}: \n"
        s += f"MC Method: {mc_method}\n"
        s += f"Num element with minimal rank ({min(per_voxel_raw_p)}) : {per_voxel_raw_p.count(min(per_voxel_raw_p))}"
        s += f"corrected {mc_method}: \n"
        s += f"{mc_method}: Out of {len(per_voxel_adjusted_p)} voxels, {len([x for x in per_voxel_adjusted_p if x < 0.05])/len(per_voxel_adjusted_p)} ({len([x for x in per_voxel_adjusted_p if x < 0.05])}) have p < 0.05\n"
        s += f"{mc_method}: {(len([x for x in per_voxel_adjusted_p if x < 0.01])/len(per_voxel_adjusted_p))} ({len([x for x in per_voxel_adjusted_p if x < 0.01])}) have p < 0.01\n"
        s += f"{mc_method}: Mean p is {np.mean(per_voxel_adjusted_p)}"
        s += f"{mc_method} significance voxels : {sum(per_voxel_adjusted_p)} / {len(per_voxel_adjusted_p)}"
        s += f"{mc_method}: min Z is {np.min(per_voxel_z)} and max {np.max(per_voxel_z)}"

        print(f"{datetime.datetime.now() } Output is : ")
        print(s)

        out_file1 = open(output_dir + start_ts + "_" + c + "_output_text.txt","w")
        out_file1.write(s)
        out_file1.close()

        print("Saved niis and str to " , output_dir)
        # return per_voxel_fdr_corrected_p_val
        plt.clf()
        plt.close()
        plt.cla()
        plt.figure()
        plt.hist(per_voxel_z, bins=100)
        plt.savefig(f"{output_dir}{start_ts}{c}_zscore_hist.png")

def analyse_spatial_isc(alignment_dict, demo, output_dir):

    print(f"{datetime.datetime.now()}: analyse_spatial_isc")
    intact_subject_ids = ["023", "030", "032", "038", "052", "079", "086", "087", "088",
                          "089", "090", "097", "098", "102", "103", "107", "109", "110"]
    scrambled_subject_ids =["088", "115", "116", "117", "118", "119", "120", "121", "122",
                            "123", "124", "125", "126", "127", "128", "129", "130", "131"]


    d = alignment_dict["aim2_fmri_data_dict"]['vodka']

    d_subject_ids = {"intact" : intact_subject_ids, "scrambled": scrambled_subject_ids}
    res_rows = []
    res_dict = {}
    ncs = [5,10,15,20]
    df_types = ["orig"] + [f"pca{n}" for n in ncs]
    for cond in ['intact','scrambled']:
        res_dict[cond] = {}
        subject_ids = d_subject_ids[cond]
        if demo:
            subject_ids = subject_ids[:2]
        for roi in ['reward','vision']:
            res_dict[cond][roi] = {}
            intact_data = d[f'vodka-{cond}-{roi}']

            num_sentences = 66
            num_perms = 35
            sen2r_dict = alignment_dict['sen2tr_dict']
            for s_i, subj_id in enumerate(subject_ids):

                res_dict[cond][roi][subj_id] = {}
                print(f"{datetime.datetime.now()}: analyse_spatial_isc ({cond},{roi}) - subj {subj_id} (#{s_i}/{len(subject_ids)})")
                other_subjects = [x for x in subject_ids if x != subj_id]
                subj_per_tr_df = intact_data[subj_id]
                subj_per_sentence_df = trdf2sendf(subj_per_tr_df, sen2r_dict )

                other_subjects_per_sentence_df = pd.concat([trdf2sendf(intact_data[s], sen2r_dict) for s in other_subjects]) \
                    .mean(level=0)

                assert (len(other_subjects_per_sentence_df) == num_sentences)
                assert (len(subj_per_sentence_df) == num_sentences)
                subj_dfs = [subj_per_sentence_df] + \
                           [pd.DataFrame(PCA(n_components=n, random_state=1).fit_transform(subj_per_sentence_df)) for n in ncs]
                other_dfs = [other_subjects_per_sentence_df] + \
                            [pd.DataFrame(PCA(n_components=n, random_state=1).fit_transform(other_subjects_per_sentence_df)) for n in ncs]
                for sentence_idx in range(num_sentences):

                    for i_df, (subj_df, other_df) in enumerate(zip(subj_dfs, other_dfs)):
                        if i_df == 0:
                            sentence_corr1 = pearsonr(subj_df.iloc[sentence_idx],
                                                      other_df.iloc[sentence_idx])
                            sentence_corr2 = spearmanr(subj_df.iloc[sentence_idx],
                                                       other_df.iloc[sentence_idx])
                            row = [df_types[i_df], cond, roi, subj_id, sentence_idx, "actual", "correlation",
                                   (np.nan, np.nan),
                                   sentence_corr1[0], sentence_corr1[1],
                                   sentence_corr2[0], sentence_corr2[1]]
                        else:
                            sentence_corr1 = distance.euclidean(subj_df.iloc[sentence_idx],
                                                                other_df.iloc[sentence_idx])
                            sentence_corr2 = distance.cosine(subj_df.iloc[sentence_idx],
                                                             other_df.iloc[sentence_idx])
                            # actual_corrs.append(sentence_corr)
                            row = [df_types[i_df], cond, roi, subj_id, sentence_idx, "actual", "distance",
                                   (np.nan, np.nan),
                                   sentence_corr1, np.nan,
                                   sentence_corr2, np.nan]
                        res_rows.append(row)

                        for r2 in range(66):
                        # random_pairs = [(random.randrange(num_sentences), random.randrange(num_sentences)) for i in range(num_perms)]
                        # for r1, r2 in random_pairs:
                            r1 = sentence_idx
                            if i_df==0:
                                new_baseline_corr1 = pearsonr(subj_df.iloc[r1],
                                                              other_df.iloc[r2])
                                new_baseline_corr2 = spearmanr(subj_df.iloc[r1],
                                                               other_df.iloc[r1])
                                row = [df_types[i_df], cond, roi, subj_id, sentence_idx, "baseline", "correlation",
                                       (r1, r2),
                                       new_baseline_corr1[0], new_baseline_corr1[1],
                                       new_baseline_corr2[0], new_baseline_corr2[1]]
                            else:
                                new_baseline_corr1 = distance.euclidean(subj_df.iloc[r1],
                                                                        other_df.iloc[r2])
                                new_baseline_corr2 = distance.cosine(subj_df.iloc[r1],
                                                                     other_df.iloc[r2])

                                row = [df_types[i_df], cond, roi, subj_id, sentence_idx, "baseline", "distance",
                                       (r1, r2),
                                       new_baseline_corr1, np.nan,
                                       new_baseline_corr2, np.nan]

                            res_rows.append(row)

    res_df = pd.DataFrame(res_rows, columns=['df_type','cond','roi','subject_id','sentence_idx','pair_type', "score_type",
                                             'idx_pairs',
                                             'score1','pval1','score2','pval2'])
    res_df.to_csv("./spatial_isc_res_df.csv", index=False)

    return res_df

def train_binary_across_subjects_perplexity_classifier(output_dir, alignment_dict):
    output_dir += "/across_subject_perplexity_classifier/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    d = alignment_dict["aim2_fmri_data_dict"]['vodka']
    nlp_df = alignment_dict["nlp_data_df"]
    nu = nlp_utils()
    nlp_df["vodka_bert_perplexity_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'bert','perplexity'))
    nlp_df["vodka_gpt_perplexity_score"] = nlp_df.vodka_text.apply(lambda x: nu.get_sequence_perplexity(x, 'gpt','perplexity'))

    sen2tr_dict = alignment_dict["sen2tr_dict"]


    res_rows = []
    ests = ALL_ESTIMATORS
    fss = ALL_FSS
    y_cols = ['vodka_bert_perplexity_score']

    total_iters = 2*2*len(fss)*len(ests)*len(y_cols)
    ii = 0

    for y_col in y_cols:
        vodka_predictability_vec = nlp_df[y_col]
        vodka_predictability_median = vodka_predictability_vec.median()
        y = vodka_predictability_vec.ge(vodka_predictability_median).tolist()
        for cond in ['intact','scrambled']:
            for roi in ['reward','vision']:

                it_key = f'vodka-{cond}-{roi}'
                cond_subjs = d[it_key].keys()
                print("Subjects are ", cond_subjs)


                X_data = [trdf2sendf(d[it_key][x], sen2tr_dict) for x in cond_subjs]
                X = pd.concat(X_data).mean(level=0)

                for est in ests:
                    for fs in fss:
                        print(f"{datetime.datetime.now()} : run {cond}, {roi}, {est}, {fs}. iter#{ii}/{total_iters}")
                        svc_ovo = make_pipeline(fs, est)
                        cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=ShuffleSplit(n_splits=1000, test_size=0.2, random_state=0))
                        res_rows.append([y_col, ii,cond,roi, est, fs, "actual", cv_scores_ovo.mean(), list(cv_scores_ovo), np.nan, np.nan])

                        ii+=1

    res_df = pd.DataFrame(data = res_rows, columns = ["y_col","ii","cond","roi","est","fs","model_type", "mean_acc", "acc_vec", "perm_actual", "perm_p"])
    ts = str(datetime.datetime.now().timestamp()).split(".")[0]
    fn = f"{output_dir}/across_subject_perplexity_classifier_{ts}.csv"
    res_df.to_csv(fn, index=False)
    print("Saved to local and to " , fn)
    return res_df


def train_multi_sentence_ordinal_regression(alignment_dict, demo, conds,
                                            output_dir):
    from sklearn.base import clone


    class OrdinalClassifier():

        def __init__(self, clf):
            self.clf = clf
            self.clfs = {}

        def fit(self, X, y):
            self.unique_class = np.sort(np.unique(y))
            if self.unique_class.shape[0] > 2:
                for i in range(self.unique_class.shape[0]-1):
                    # for each k - 1 ordinal value we fit a binary classification problem
                    binary_y = (y > self.unique_class[i]).astype(np.uint8)
                    clf = clone(self.clf)
                    clf.fit(X, binary_y)
                    self.clfs[i] = clf

        def predict_proba(self, X):
            clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
            predicted = []
            for i,y in enumerate(self.unique_class):
                if i == 0:
                    predicted.append(1 - clfs_predict[y][:,1])
                elif y in clfs_predict:
                    predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
                else:
                    predicted.append(clfs_predict[y-1][:,1])
            return np.vstack(predicted).T

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    output_dir += "/intact_perplexity_ordinal_regression/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv = 2 if demo else 5
    d = alignment_dict["aim2_fmri_data_dict"]['vodka']
    nlp_df = alignment_dict["nlp_data_df"]
    sen2tr_dict = alignment_dict["sen2tr_dict"]
    vodka_perplexity_vec = nlp_df['vodka_perplexity_score']
    vodka_perplexity_ranks = scipy.stats.rankdata(vodka_perplexity_vec)
    vodka_perplexity_ranks = [x-1 for x in vodka_perplexity_ranks]
    ydf = pd.DataFrame()
    ydf["y"] = [str(x) for x in vodka_perplexity_ranks]
    ydf["y"] = ydf["y"].astype('category')
    res_rows = []
    from sklearn.linear_model import HuberRegressor, BayesianRidge, RANSACRegressor, TheilSenRegressor
    methods3 = [RANSACRegressor(min_samples=21), RANSACRegressor(min_samples=10), RANSACRegressor(min_samples=5),
                RANSACRegressor(min_samples=30), HuberRegressor(), BayesianRidge(), TheilSenRegressor(),
                ]
    # methods3 = []
    methods1 = ALL_ESTIMATORS
    # methods1 = [SVC(kernel='poly', probability=True)]
    methods1 = []
    methods2 = ['newton','bfgs','lbfgs','powell','cg','ncg','basinhopping']
    methods2 = []
    methods = methods3 + methods2 + methods1
    fss = ALL_FSS
    ii = 0
    for cond in ['intact','scrambled']:
        for roi in ['reward','vision']:

            it_key = f'vodka-{cond}-{roi}'
            intact_subjs = d[it_key].keys()
            print("Subjects are ", intact_subjs)
            for i, loo_subj in enumerate(intact_subjs):
                train_subjects = [x for x in intact_subjs if x != loo_subj]
                X_data = [trdf2sendf(d[it_key][x], sen2tr_dict) for x in train_subjects]
                X_train = pd.concat(X_data).mean(level=0)
                # y_train = [str(x) for x in range(len(X_train))]

                X_test = trdf2sendf(d[it_key][loo_subj], sen2tr_dict)
                # y_test = [str(x) for x in range(len(X_test))]
                for fs in fss:

                    try:
                        fs_fitted = fs.fit(X_train)

                    except ValueError as e:
                        print("Error - ", e)
                    except TypeError:
                        fs_fitted = fs.fit(X_train, vodka_perplexity_ranks)

                    for mt in methods3:
                        print(f"{datetime.datetime.now()}: run {roi},{cond}, {loo_subj}, {fs}, {mt} : iter {ii}/{1*2 *len(methods)* len(intact_subjs) * len(fss)}")

                        mt.fit(X = X_train, y = vodka_perplexity_ranks) #shape here is 66, 12031
                        y_pred = mt.predict(X_test)
                        pred_mse = mean_squared_error(y_pred, vodka_perplexity_ranks)
                        acc = mt.score(X_test, vodka_perplexity_vec)
                        acc1 = median_absolute_error(y_pred, vodka_perplexity_vec)
                        acc2 = explained_variance_score(y_pred, vodka_perplexity_vec)
                        acc3 = r2_score(y_pred, vodka_perplexity_vec)
                        res_rows.append([np.nan, ii,cond,roi, loo_subj, fs, "regression_model", mt, pred_mse, acc,
                                         acc1, acc2, acc3, list(y_pred), list(vodka_perplexity_ranks)])
                        ii+=1

                    for mt in methods1:
                        print(f"{datetime.datetime.now()}: run {roi},{cond}, {loo_subj}, {fs}, {mt} : iter {ii}/{2*2 *len(methods)* len(intact_subjs) * len(fss)}")

                        clf = OrdinalClassifier(mt)
                        clf.fit(fs_fitted.transform(X_train), vodka_perplexity_ranks)
                        y_pred = clf.predict(fs_fitted.transform(X_test))
                        pred_mse = mean_squared_error(y_pred, vodka_perplexity_ranks)
                        acc = (y_pred == vodka_perplexity_ranks).mean()

                        res_rows.append([np.nan, ii,cond,roi, loo_subj, fs, "ordered_classes", mt, pred_mse, acc, list(y_pred), list(vodka_perplexity_ranks)])

                        ii+=1
                        for q in range(1):
                            y_train_shuffled = sorted(vodka_perplexity_ranks, key=lambda k: random.random())
                            clf = OrdinalClassifier(mt)
                            clf.fit(X_train, y_train_shuffled)
                            y_pred = clf.predict(X_test)
                            acc = (y_pred == y_train_shuffled).mean()
                            acc1 = median_absolute_error(y_pred, y_train_shuffled)
                            acc2 = explained_variance_score(y_pred, y_train_shuffled)
                            acc3 = r2_score(y_pred, y_train_shuffled)
                            pred_mse = mean_squared_error(y_pred, y_train_shuffled)
                            row2 = [q,ii,cond,roi, loo_subj, fs, "shuffled_labeld", mt, pred_mse, acc,
                                    acc1, acc2, acc3, list(y_pred), list(y_train_shuffled)]
                            res_rows.append(row2)


    res_df = pd.DataFrame(data = res_rows, columns = ["q","ii","cond","roi","loo_subj","fs_method","ord_method", "fit_method",
                                                      "mse","acc", "acc1", "acc2", "acc3", "y_pred","y_true"])
    ts = str(datetime.datetime.now().timestamp()).split(".")[0]
    fn = f"{output_dir}/intact_perplexity_ordinal_regression_{ts}.csv"
    res_df.to_csv(fn, index=False)
    print("Saved to to " , fn)
    return res_df


def get_alignment_dict(load_existing = True, save = False):

    if load_existing :
        from configs.params import alignment_dict_fn_onlyaim2 as alignment_dict_fn

        print(f"{datetime.datetime.now()} Load dict from {alignment_dict_fn}")
        alignment_dict = joblib.load(alignment_dict_fn)
        print(f"{datetime.datetime.now()}: Dict is loaded")
    else:
        from main import create_dict
        alignment_dict = create_dict(use_existing_aligned_df = False, save = save)
        print(f"{datetime.datetime.now} : Return with created dict 1")
    return alignment_dict

def analyse_dict(alignment_dict,analyse_brain_sig,
                 encoding_model,
                 spatial_isc,
                 multi_sentence_classifier,
                 cross_subject_perplexity_classifier,
                 train_subject_condition_classifier,
                 within_subject_perplexity_classifier,
                 analyse_brain_sig_wb,
                 multi_sentence_ordinal_regression,
                 cond_classifier_wb_searchlight,
                 across_subjects_binary_perplexity_classifier,
                 out_dir_suf = "",
                 demo = False):

    output_dir = f"{aim2_analysis_artifacts_dir}/120422_Newalignment_dict{out_dir_suf}/"
    print(f"output dir is {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"The new output directory is created!")
    # alignment_dict['nlp_data_df'].to_csv(output_dir + "/nlp_data_df.csv")

    if within_subject_perplexity_classifier:
        print(f"{datetime.datetime.now()} : Training within_subject_perplexity_classifier")
        wsc_res_df = train_within_subject_perplexity_classifier(alignment_dict, demo, conds = ['reward','vision'],
                                                                output_dir = output_dir)
        wsc_res_df.to_csv(output_dir + "wsc_res_df.csv", index=False)
        print(f"{datetime.datetime.now()} : Saved within_subject_perplexity_classifier to {output_dir}")

    if across_subjects_binary_perplexity_classifier:
        print(f"{datetime.datetime.now()} : Training train_binary_across_subjects_perplexity_classifier")
        train_binary_across_subjects_perplexity_classifier(alignment_dict = alignment_dict, output_dir = output_dir)

    if multi_sentence_ordinal_regression:
        print(f"{datetime.datetime.now()} : Training multi_sentence_ordinal_regression")
        train_multi_sentence_ordinal_regression(alignment_dict, demo = demo, conds=['reward','vision'],
                                                output_dir = output_dir)
    if encoding_model:
        print(f"{datetime.datetime.now()} : Training encoding_model")
        train_encoding_model(alignment_dict, demo = False, conds=['reward','vision'],
                             output_dir = output_dir)
    if multi_sentence_classifier:
        print(f"{datetime.datetime.now()} : Training multi_sentence_classifier")
        multi_sentences_classifier_res_df = train_multi_sentence_classifier(alignment_dict, demo = False, conds=['reward','vision'],
                                        output_dir = output_dir)
        multi_sentences_classifier_res_df.to_csv(output_dir + "multi_sentences_classifier_res_df.csv", index=False)

    if spatial_isc:
        print(f"{datetime.datetime.now()} : Training spatial_isc")
        si_res_df = analyse_spatial_isc(alignment_dict, demo = False,
                                                                output_dir = output_dir)
        si_res_df.to_csv(output_dir + "si_res_df.csv", index=False)
        print(f"{datetime.datetime.now()} : Saved spatial_isc results to {output_dir}")


    if cross_subject_perplexity_classifier:
        print(f"{datetime.datetime.now()} : Training cross_subject_perplexity_classifier")
        cspc_res_df = train_cross_subject_perplexity_classifier(alignment_dict, demo, conds = ['reward','vision'],
                                                                output_dir = output_dir)
        print(f"{datetime.datetime.now()} : Saved cross_subject_perplexity_classifier to {output_dir}")
        cspc_res_df.to_csv(output_dir + "cspc_res_df.csv", index=False)

    if train_subject_condition_classifier:
        print(f"{datetime.datetime.now()} : Training train_subject_condition_classifier")
        scc_res_df = train_subject_cond_classifier(alignment_dict, demo, conds = ['reward','vision'], level='subject', out_dir = output_dir)
        scc_res_df.to_csv(output_dir + "scc_res_df.csv", index=False)



    if analyse_brain_sig:
        from configs.params import alignment_dict_fn_onlyaim2\
        for c in ['reward','vision']:
            analyse_brain(output_dir = output_dir , c=c,  num_iters = 1000, use_existing=True, save = False,
                          alignment_dict_fn= alignment_dict_fn_onlyaim2)#, alignment_dict = alignment_dict)

    if analyse_brain_sig_wb:
        from configs.params import alignment_dict_fn_onlyaim_wb
        analyse_brain(output_dir = output_dir , c='wholebrain',  num_iters = 1000,
                      use_existing = True, save=False, alignment_dict_fn= alignment_dict_fn_onlyaim_wb)

    print("output_dir is ", output_dir)

def main():
    print(f"{datetime.datetime.now()} : Start  Main")
    demo = False

    alignment_dict = get_alignment_dict(load_existing=True, save=False)
    if len(alignment_dict["tr2sentences_dict"]) == 0:
        sentence2tr_dict_fn = ARTIFACTS_DIR + "sent2tr_dict.jbl"
        sen2tr_dict = joblib.load(sentence2tr_dict_fn)
        alignment_dict["sen2tr_dict"] = sen2tr_dict

    print(f"{datetime.datetime.now} : Return with created dict 2")
    analyse_dict(alignment_dict, demo=True,
                 encoding_model = False,
                 multi_sentence_classifier = True,
                 spatial_isc = True,
                 analyse_brain_sig = False,
                 cross_subject_perplexity_classifier = True,
                 train_subject_condition_classifier = True,
                 within_subject_perplexity_classifier = True,
                 analyse_brain_sig_wb = True,
                 multi_sentence_ordinal_regression = True,
                 cond_classifier_wb_searchlight = True,
                 across_subjects_binary_perplexity_classifier = True,
                 out_dir_suf="_Try_Final_code_before_github",# wb_analysis_1000iters_",
                 # out_dir_suf="_vmpfc_z_p_FINAL_for_yaara"
                 )
    print(f"{datetime.datetime.now()} : Done")
if __name__ == '__main__':
    main()

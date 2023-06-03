from pylab import *
import scipy.stats
import pandas as pd
import seaborn as sns
from palettable.colorbrewer.qualitative import Set2_7

from itertools import zip_longest, product

def xplode(df, explode, zipped=True):
    method = zip_longest if zipped else product

    rest = {*df} - {*explode}

    zipped = zip(zip(*map(df.get, rest)), zip(*map(df.get, explode)))
    tups = [tup + exploded
            for tup, pre in zipped
            for exploded in method(*pre)]

    return pd.DataFrame(tups, columns=[*rest, *explode])[[*df]]


def strip_fs(fs_col):
    return [x.replace("n_components=","").replace("Fast","").replace("Select","").replace("KBest","K") for x in fs_col]

def strip_est(est_col):
    return [x.replace("n_estimators=","").replace("random_state=7","").replace(", ","").replace("solver=","").replace("penalty=","").replace("n_neighbors=","").replace("probability=True","").replace("max_iter=1000","").replace("Classifier","").replace("RandomForest","RF").replace("KNeighbors","KN").replace("LogisticRegression","LR").replace("random_state=0","").replace("kernel=","")
                               for x in est_col]

def strip_catplot_title(g):
    [plt.setp(ax.texts, text="") for ax in g.axes.flat] # remove the original texts
    # important to add this before setting titles
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    return g

import binsreg

def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)

    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})

    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']

    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est


def reload_str():
    """%load_ext autoreload
    %autoreload 2
    """
def draw_stars_box(regr_acc_l, rand_acc_l, stat_test, suf,ir = None, show=True, ax=None, reverse_labels = False, alternative='two-sided',
                   verbose = True, ylabel = "CORRELATION", title = "",xlabels = ['Actual','Shuffled'], add_fig = True):
    def stars(p):
        if p < 0.0001:
            return "****"
        elif (p < 0.001):
            return "***"
        elif (p < 0.01):
            return "**"
        elif (p < 0.05):
            return "*"
        else:
            return "-"
    # print("Hwwlp")
    #https://github.com/jbmouret/matplotlib_for_papers
    if stat_test == 'wilcoxon':
        z, p = scipy.stats.wilcoxon(regr_acc_l, rand_acc_l, alternative=alternative)
    if stat_test == 'mannwhittney':
        z, p = scipy.stats.mannwhitneyu(regr_acc_l, rand_acc_l, alternative=alternative)
    p_value = p * 2
    stars_txt = stars(p)
    s = f"{suf} : {stat_test}: z = {z} , p = {p} "
    if verbose:
        print(s)

    if add_fig:
        fig = figure(figsize=(3,3))
    if ax is None:
        ax = fig.add_subplot(111)

    bp = ax.boxplot([regr_acc_l, rand_acc_l])
    params = {
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [5, 8]
    }
    rcParams.update(params)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out', size=15)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    # colors, as before
    colors = Set2_7.mpl_colors

    for i in range(0, len(bp['boxes'])):
        bp['boxes'][i].set_color(colors[i])
        # we have two whiskers!
        bp['whiskers'][i*2].set_color(colors[i])
        bp['whiskers'][i*2 + 1].set_color(colors[i])
        bp['whiskers'][i*2].set_linewidth(2)
        bp['whiskers'][i*2 + 1].set_linewidth(2)
        # fliers
        # (set allows us to set many parameters at once)
        bp['fliers'][i].set(markerfacecolor=colors[i],
                            marker='o', alpha=0.75, markersize=6,
                            markeredgecolor='none')
        bp['medians'][i].set_color('black')
        bp['medians'][i].set_linewidth(3)
        # and 4 caps to remove
        for c in bp['caps']:
            c.set_linewidth(0)
    for i in range(len(bp['boxes'])):
        box = bp['boxes'][i]
        box.set_linewidth(0)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX,boxY))
            boxPolygon = Polygon(boxCoords, facecolor = colors[i], linewidth=0)
            ax.add_patch(boxPolygon)

    if reverse_labels:
        xlabels.reverse()
    ax.set_xticklabels(xlabels, fontsize=15)


    y_max = np.max(np.concatenate((regr_acc_l, rand_acc_l)))
    y_min = np.min(np.concatenate((regr_acc_l, rand_acc_l)))

    ax.annotate("", xy=(1, y_max), xycoords='data',
                xytext=(2, y_max), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                                connectionstyle="bar,fraction=0.2"))
    ax.text(1.5, y_max + abs(y_max - y_min)*0.2, stars_txt,
            horizontalalignment='center',
            verticalalignment='center')
    if ir is not None:
        ax.text(0.6, 0.88, ["I","II"][ir], fontsize=18)
        ax.set_ylim([0,1])
    # ax.set_suptitle(f"Per iteration accuracy {suf} - {p}", fontsize = 12)
    # ax.subplots_adjust(left=0.2)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title)
    if show:
        plt.show()


    return s, ax

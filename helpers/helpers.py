import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import seaborn as sns
sns.set_style("whitegrid")
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def cp_split(x_train, y_train, calib_size=0.33):
    x_prop, x_calib, y_prop, y_calib = train_test_split(x_train, y_train, test_size=calib_size, shuffle=True, stratify=y_train, random_state=42)
    print("Prop:", x_prop.shape, y_prop.shape, "\nCalib:", x_calib.shape, y_calib.shape)
    return x_prop, y_prop, x_calib, y_calib


def get_label_conditional_knns(x_prop, y_prop):
    # NCM in neighbor_distance calculated as distance to same class (because y_prop==l)
    dists_knns = dict()
    labels_unique = np.unique(y_prop)
    for l in labels_unique:
        dists_knns[l] = KNeighborsClassifier().fit(x_prop[y_prop==l], np.ones((x_prop[y_prop==l].shape[0])))
    return dists_knns


def neighbor_distance(xs_test, knn, k=5):
    dists = knn.kneighbors(xs_test, n_neighbors=k, return_distance=True)[0].sum(axis=1)
    return dists


def get_ncm_scores(xs, knns, K):
    xs_scores = dict()
    for i in tqdm(list(knns.keys()), desc="NCMs for labels"):
        xs_scores[i] = neighbor_distance(xs, knns[i], K)

    print(list(xs_scores.keys()))
    print(xs_scores[0].shape)
    return xs_scores


def calc_pvalues(calib_ncms, test_ncms):
    calib_ncms.sort()  # ascending, required by np.searchsorted()
    js = np.searchsorted(calib_ncms, test_ncms, side='left') # find sorted index
    rank = len(calib_ncms) - js + 1  # | alpha test <= alpha calib| + 1
    ps = rank / (len(calib_ncms)+1)  # +1 because calib extended by test sample
    return ps


def get_pvalues(calib_ncms, test_ncms):
    # should this be against calib ncms of the right class?
    pvalues = dict()
    for i in tqdm(list(calib_ncms.keys()), desc="p-values for labels"):
        pvalues[i] = calc_pvalues(calib_ncms[i], test_ncms[i])

    print(list(pvalues.keys()))
    print(pvalues[0].shape)
    return pvalues


def show_confidence_feature_space(epsilon, grid_xs, grid_ys, ps_grid_target, x_train_source=None, y_train_source=None, x_train_target=None, y_train_target=None, title=None, target_name="", source_name=""):
    cols = 3
    labels = list(ps_grid_target.keys())
    rows = math.ceil(len(labels)/cols)
    figsize = (20, 5*rows)
    fig, axs = plt.subplots(rows, cols, sharey=True, sharex=True, figsize=figsize)
    fig.set_tight_layout(True)
    try:
        sns.scatterplot(x=x_train_target[:, 0], y=x_train_target[:, 1], s=5, hue=y_train_target, palette="hls", ax=axs[0][0])
    except:  # if rows==1
        sns.scatterplot(x=x_train_target[:, 0], y=x_train_target[:, 1], s=5, hue=y_train_target, palette="hls", ax=axs[0])

    for i, l in enumerate(labels):
        ax_x = math.floor((i+1)/cols)
        ax_y = (i+1)%cols

        try:
            axs[ax_x, ax_y].set_title("Class {}".format(i))
            axs[ax_x, ax_y].contourf(grid_xs, grid_ys, (ps_grid_target[i]>epsilon).reshape(len(grid_ys), len(grid_xs), order='F'))
            if x_train_target.any():
                sns.scatterplot(x=x_train_target[:, 0][y_train_target==l], y=x_train_target[:, 1][y_train_target==l], s=10, color='green', ax=axs[ax_x][ax_y], alpha=0.6, label=target_name)
            if x_train_source.any():
                sns.scatterplot(x=x_train_source[:, 0][y_train_source==l], y=x_train_source[:, 1][y_train_source==l], s=10, color='red', ax=axs[ax_x][ax_y], alpha=0.6, label=source_name)
        except:  # if rows==1
            axs[ax_y].set_title("Class {}".format(i))
            axs[ax_y].contourf(grid_xs, grid_ys, (ps_grid_target[i]>epsilon).reshape(len(grid_ys), len(grid_xs), order='F'))
            if x_train_target.any():
                sns.scatterplot(x=x_train_target[:, 0][y_train_target==l], y=x_train_target[:, 1][y_train_target==l], s=10, color='green', ax=axs[ax_y], alpha=0.6, label=target_name)
            if x_train_source.any():
                sns.scatterplot(x=x_train_source[:, 0][y_train_source==l], y=x_train_source[:, 1][y_train_source==l], s=10, color='red', ax=axs[ax_y], alpha=0.6, label=source_name)

    if title:  plt.suptitle(title)
    if target_name!="" or source_name!="":  plt.legend()
    plt.show()


def cartesian(arrays, out=None):
    # https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    print(m)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

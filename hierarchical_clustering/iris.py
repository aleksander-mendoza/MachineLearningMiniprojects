import os
import torchvision as tv
from matplotlib.gridspec import GridSpec
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

SAMPLE_COUNT = 128

iris = load_iris()
X = iris['data']
y = iris['target']

embed2d = TSNE(n_components=2).fit_transform(X)
fig, ax = plt.subplots()
cluster = AgglomerativeClustering(compute_full_tree=True, n_clusters=3, affinity='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(embed2d)
sc = ax.scatter(embed2d[:, 0], embed2d[:, 1], c=cluster_labels)
for i, (label, position) in enumerate(zip(y, embed2d)):
    ax.annotate(str(label), position)

annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = " ".join([str(y[n]) for n in ind["ind"]])
    annot.set_text(text)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()

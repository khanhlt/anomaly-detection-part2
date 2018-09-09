from load_data import read_image_and_label
import numpy as np
from sklearn.preprocessing import StandardScaler

train, name = read_image_and_label("../dataset/ok_data")
labels = [0] * len(train)

label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

label_ids = np.array([label_to_id_dict[x] for x in labels])

import matplotlib.pyplot as plt


def visualize_scatter(data_2d, label_ids, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.savefig('scatter.png')
    plt.show()


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=3)
pca_result = pca.fit_transform(train)
tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(pca_result)
re = StandardScaler().fit_transform(tsne_result)
visualize_scatter(re, label_ids)

np.savetxt('tsne_result.txt', re, fmt='%f')
np.savetxt('image_name.txt', name, fmt='%s')


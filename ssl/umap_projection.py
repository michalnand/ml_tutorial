import matplotlib.pyplot as plt

import numpy
import umap
from sklearn.manifold import TSNE


def umap_projection(x, y):

    umap_model = umap.UMAP(n_components=2) 

    # Fit and transform the data to reduce to 2 dimensions
    z_proj = umap_model.fit_transform(x)


    colormap = numpy.array(["red","blue","green","yellow","gray","magenta","orange","purple","black","brown"])

    plt.scatter(z_proj[:, 0], z_proj[:, 1], c=colormap[y])
    plt.show()




def tsne_projection(x, y):
    # Fit and transform the data to reduce to 2 dimensions
    z_proj = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x)

    colormap = numpy.array(["red","blue","green","yellow","gray","magenta","orange","purple","black","brown"])

    plt.scatter(z_proj[:, 0], z_proj[:, 1], c=colormap[y])
    plt.show()
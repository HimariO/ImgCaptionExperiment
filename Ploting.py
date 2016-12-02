# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# We import seaborn to make nice plots.
import seaborn as sns

import json
import os
from PIL import Image

# Random state.
RS = 20150101
annotation_path = './annotations/captions_train2014.json'

with open(annotation_path, 'r') as json_file:
    anno = json.loads(json_file.read())

# image_paths = list(map(lambda img: './train2014/' + img['file_name'], anno['images']))
# image_ids = list(map(lambda img: img['id'], anno['images']))
image_paths = ['./PreProcess/' + i for i in os.listdir('./PreProcess/') if os.path.isfile('./PreProcess/' + i) and i[-3:] == 'jpg']


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure()
    ax = plt.subplot()
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[1])
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    ax.axis('off')
    ax.axis('tight')

    img_boxs = []
    for ind, point in zip(range(len(x)), x):
        oImg = OffsetImage(plt.imread(image_paths[ind]), zoom=.2)
        ab = AnnotationBbox(oImg, xy=(point[0], point[1]), xycoords='data', boxcoords="offset points")
        img_boxs.append(ax.add_artist(oImg))
        print('ImgBox[%d]' % ind)

    return f, ax, sc, img_boxs


def scatter_PIL(p, size=(1000, 1000)):
    # create a white background.
    base = Image.new('RGB', size, color=1)

    x_max = max(p, key=lambda _p: _p[0])[0]
    y_max = max(p, key=lambda _p: _p[1])[1]

    x_min = min(p, key=lambda _p: _p[0])[0]
    y_min = min(p, key=lambda _p: _p[1])[1]

    # resize_scaler = max([x_max - x_min, y_max - y_min]) / size[0]
    resize_scaler = 150
    center_offset = ((x_max + x_min) / 2, (y_max + y_min) / 2)

    for i in range(len(p)):
        p[i][0] -= center_offset[0]
        p[i][0] *= resize_scaler
        p[i][1] -= center_offset[1]
        p[i][1] *= resize_scaler

    for ind, point in zip(range(len(p)), p):
        oImg = Image.open(image_paths[ind])
        _img = oImg.resize((int(oImg.size[0] * 0.2), int(oImg.size[1] * 0.2)))

        new_pos = (int(size[0] / 2 + point[0]), int(size[1] / 2 + point[1]))
        base.paste(_img, new_pos)

    base.save('./Plot/Scatter.jpg')

tsne_proj = np.load('./data/t_sne_video_.npy')

print('Ploting...')
# scatter(tsne_proj[:100], [])
scatter_PIL(tsne_proj, size=(5000, 5000))
# plt.savefig('./Plot/digits_tsne-generated.png', dpi=120)

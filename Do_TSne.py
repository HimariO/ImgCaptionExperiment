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

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import json
import math

# Random state.
RS = 20150101

start_img = 1000
end_img = 2000

# name_tag = str(int(start_img / 1000)) + 'k_' + str(math.ceil(end_img / 1000)) + 'k'
name_tag = '_video_'
feat_path = './data/feats' + name_tag + '.npy'
ids_path = './data/IDs' + name_tag + '.npy'

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

feats = np.load(feat_path)
ids = np.load(ids_path)
feats_flat = []


for feat in feats:
    feats_flat.append(feat.reshape([512 * 14 * 14]))

print('Start TSNE...')
tsne_proj = TSNE(random_state=RS).fit_transform(feats_flat)

print('Saving result.')
np.save('./data/t_sne' + name_tag + '.npy', tsne_proj)

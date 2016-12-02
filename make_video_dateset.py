import numpy as np
import os
import tensorflow as tf

from VGG.vgg19 import Vgg19
import VGG.utils as utils
import json
import math
from termcolor import colored


image_path = './PreProcess/'
name_tag = '_video_'
feat_path = './data/feats' + name_tag + '.npy'

image_paths = [image_path + i for i in os.listdir(image_path) if os.path.isfile(image_path + i) and i[-3:] == 'jpg']
image_ids = []
image_number = len(image_paths)

feats = np.zeros([image_number] + [512, 14, 14])
ids = np.zeros([image_number])

with tf.Session() as sess:
    vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
    image_holder = tf.placeholder('float', [1, 224, 224, 3])
    vgg.build(image_holder)

    for ind, path in zip(range(image_number), image_paths):
        print('Loading: %d' % (ind), colored(path, color='yellow'))

        try:
            img = utils.load_image(path)
            batch = img.reshape((1, 224, 224, 3))

            # get features of shape [num of img, 14, 14, 512]
            feature5_3 = sess.run([vgg.conv5_3], feed_dict={image_holder: batch})
            feats[ind] = np.reshape(feature5_3[0], [512, 14, 14])
            ids[ind] = image_ids[ind]
            del img, batch
        except:
            # some image are not rgb format, so can't be reshape.
            ids[ind] = -1

print('Start Saving...')
np.save('./data/IDs' + name_tag + '.npy', ids)
np.save(feat_path, feats)

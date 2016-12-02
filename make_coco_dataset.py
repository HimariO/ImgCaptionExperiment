import numpy as np
import os
import tensorflow as tf

from VGG.vgg19 import Vgg19
import VGG.utils as utils
import json
import math
from termcolor import colored

start_img = 1000
end_img = 2000

annotation_path = './annotations/captions_train2014.json'
image_path = './train2014/'
annotation_result_path = './data/annotations.pickle'
name_tag = str(int(start_img / 1000)) + 'k_' + str(math.ceil(end_img / 1000)) + 'k'
feat_path = './data/feats' + name_tag + '.npy'
anno = None

with open(annotation_path, 'r') as json_file:
    anno = json.loads(json_file.read())

image_paths = list(map(lambda img: image_path + img['file_name'], anno['images']))
image_ids = list(map(lambda img: img['id'], anno['images']))
image_number = end_img - start_img
# image_number = 100

feats = np.zeros([image_number] + [512, 14, 14])
ids = np.zeros([image_number])

with tf.Session() as sess:
    vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
    image_holder = tf.placeholder('float', [1, 224, 224, 3])
    vgg.build(image_holder)

    for ind, path in zip(range(start_img, end_img), image_paths[start_img: end_img]):
        print('Loading: %d' % (ind), colored(path, color='yellow'))

        try:
            img = utils.load_image(path)
            batch = img.reshape((1, 224, 224, 3))

            # get features of shape [num of img, 14, 14, 512]
            feature5_3 = sess.run([vgg.conv5_3], feed_dict={image_holder: batch})
            feats[ind - start_img] = np.reshape(feature5_3[0], [512, 14, 14])
            ids[ind - start_img] = image_ids[ind]
            del img, batch
        except:
            # some image are not rgb format, so can't be reshape.
            ids[ind - start_img] = -1

print('Start Saving...')
np.save('./data/IDs' + name_tag + '.npy', ids)
np.save(feat_path, feats)

import numpy as np
import os
import tensorflow as tf

from VGG.vgg19 import Vgg19
import VGG.utils as utils
import json
import math
from termcolor import colored


class CocoDataSet(object):

    def __init__(self,
                 batch_size=100,
                 annotation_path='./annotations/captions_train2014.json',
                 image_path='./train2014/'):
        self.annotation_path = annotation_path
        self.image_path = image_path

        self.image_num = 0
        self.image_paths = []
        self.image_ids = []

        self.current_id = 0
        self.batch_size = batch_size

    def __iter__(self):
        with open(self.annotation_path, 'r') as json_file:
            self.anno = json.loads(json_file.read())
            self.anno['images'] = sorted(self.anno['images'], key=lambda x: x['id'])
            self.anno['annotations'] = sorted(self.anno['annotations'], key=lambda x: x['image_id'])
            self.current_id = 0

            self.image_num = len(self.anno['images'])
            self.image_paths = list(map(lambda img: self.image_path + img['file_name'], self.anno['images']))
            self.image_ids = list(map(lambda img: img['id'], self.anno['images']))
            # self.sess = tf.Session()
            # self.vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
        return self

    def __next__(self):
        if self.current_id < self.image_num:
            end_num = self.current_id + self.batch_size if self.current_id + self.batch_size < self.image_num else self.image_num
            (feats, captions, ids) = self.GetFeats(start_img=self.current_id, end_img=end_num)
            self.current_id += self.batch_size
            return (feats, captions, ids)
        else:
            raise StopIteration

    def GetFeats(self, start_img=0, end_img=2000, save_npy=False):
        """
        Get Feature from vgg19 with more than one image input.
        if you dont want to save features as npy file, you can iter through this object directal.
        ex: [batch_feat_cap_id for batch_feat_cap_id in CocoDataSet]
        """

        # don't need this for now, since i make this a function.
        # start_img = 1000
        # end_img = 2000

        name_tag = str(int(start_img / 1000)) + 'k_' + str(math.ceil(end_img / 1000)) + 'k'
        feat_path = './data/feats' + name_tag + '.npy'
        image_number = end_img - start_img
        # image_number = 100

        feats = np.zeros([image_number] + [512, 14, 14])
        ids = np.zeros([image_number])
        captions = []

        # if self.sess is not None:
        with tf.Session() as sess:
            vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
            image_holder = tf.placeholder('float', [1, 224, 224, 3])
            vgg.build(image_holder)

            for ind, path in zip(range(start_img, end_img), self.image_paths[start_img: end_img]):
                print('Loading: %d' % (ind), colored(path, color='yellow'))

                try:
                    img = utils.load_image(path)
                    batch = img.reshape((1, 224, 224, 3))

                    # get features of shape [num of img, 14, 14, 512]
                    feature5_3 = sess.run([vgg.conv5_3], feed_dict={image_holder: batch})

                    feats[ind - start_img] = np.reshape(feature5_3[0], [512, 14, 14])
                    ids[ind - start_img] = self.image_ids[ind]
                    captions.append(list(map(lambda x: x['caption'], self.anno['annotations'][ind: ind + 5])))

                    del img, batch
                except:
                    # some image are not rgb format, so can't be reshape.
                    ids[ind - start_img] = -1
                    captions.append([])

        if save_npy:
            print('Start Saving...')
            np.save('./data/IDs' + name_tag + '.npy', ids)
            np.save(feat_path, feats)

        return (feats, captions, ids)

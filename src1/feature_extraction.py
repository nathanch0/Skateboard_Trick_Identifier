import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

"""
Credit to Kernix blog
"""

def images_list(image_directory):
    image_list = [image_directory+f for f in os.listdir(images_directory) if re.search('jpg|JPG', f)]
    return image_list

def create_graph():
    """
    create_graph loads the inception model to memory, should be called before
    calling extraction.

    model_path: path to inception model in protobuf form.
    """
    model_dir = 'imagenet'
    with gfile.FastGFile(os.path.join(model_dir,
                    'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extraction(list_images):
    """
    extract_features computed the inception bottleneck feature for a list of images

    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for ind, image in enumerate(list_images):
        if (ind%100 == 0):
            print('Processing %s...' % (image))
        if not gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)

    image_data = gfile.FastGFile(image, 'rb').read()

    predictions = sess.run(next_to_last_tensor,
                        {'DecodeJpeg/contents:0': image_data})

    features[ind,:] = np.squeeze(predictions)

    labels.append(re.split('_\d+',image.split('/')[1])[0])

    return features

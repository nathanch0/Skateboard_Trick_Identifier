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

def image_list(image_dir):
    """Builds a list of training images from the file system.
      Analyzes the sub folders in the image directory, and returns a data structure
      describing the lists of images for each label and their paths.
      Args:
        image_dir: String path to a folder containing subfolders of images.

      Returns:
        A dictionary containing an entry for each label subfolder
      """
    # Checks if the directory name even exists!
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)] # This will create a list of sub directories i.e Kickflip, ollie

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False # Because the first element is the root directory, we skip it to go into the sub directories
            continue
        extensions = ['jpg', 'JPG'] # Image extension
        file_list = [] # File path list of all the images in the directory
        dir_name = os.path.basename(sub_dir) # This will make the dir_name to be 'Kickflip' or 'Ollie'
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension) # Making a file path for all photos with given extension
            file_list.extend(gfile.Glob(file_glob)) # This will add the 'file_glob' string to file_list
        if not file_list:
            print('No files have been found')
            continue
        if len(file_list) < 20: # This will check the length of the file_list
            print('There is less than 20 photos in this directory! There may not be enough pictures!')

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower()) #This makes the label name for each spacific Image
        result[label_name] = {'dir':dir_name,
                             'train':file_list}
    return result


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

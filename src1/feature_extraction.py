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
Initially run the image_list function first to establish a list of all image paths
in training directory. Then run the extraction function to populate the features and
labels list.
"""

def image_list(image_dir, type_input='list'):
    """
      Builds a list of training images from the file system.
      Analyzes the sub folders in the image directory, and returns a data structure
      describing the lists of images for each label and their file paths.

      Args:
        image_dir: String path to a folder containing subfolders of images.
        type_input: This string will determine what is returned

      Returns:
        A dictionary containing an entry for each label subfolder
      """
    # Checks if the directory name even exists!
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)] # This will create a list of sub directories i.e Kickflip, ollie
    final_result = []

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

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower()) #This makes the label name for each spacific Image

        result[label_name] = file_list

    if type_input == 'dict':
        return result
    else:
        for value in result.values():
            final_result.extend(value)
        return final_result

"""
Credit to Kernix blog
"""

def create_graph():
    """
    create_graph loads the inception model to memory, should be called before
    calling extraction. This is called in the extraction function.
    """
    model_dir = 'imagenet'
    with gfile.FastGFile(os.path.join('..',model_dir,
                    'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extraction(final_result):
    """
    extract_features computed the inception bottleneck feature for a list of images
    using tensorflow.
    Args:
        final_result: array of image path
    Returns:
        2-d array in the shape of (len(image_paths), 2048)
    """

    nb_features = 2048
    features = np.empty((len(final_result),nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(final_result):
            if (ind%200 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

                image_data = gfile.FastGFile(image, 'rb').read()

                predictions = sess.run(next_to_last_tensor,
                            {'DecodeJpeg/contents:0': image_data})

                features[ind,:] = np.squeeze(predictions)

                labels.append(re.split('_\d+',image.split('/')[1])[0])


    if len(labels) > 1:
        model_output_path = 'pickle_files/'
        with open(model_output_path + 'features.pkl','wb') as f:
            pickle.dump(features, f)

        with open(model_output_path + 'labels.pkl', 'wb') as l:
            pickle.dump(labels, l)
        print('Extraction is completed. Please train the model now!')
    else:
        return features

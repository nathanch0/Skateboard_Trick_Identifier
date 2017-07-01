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
    image_list = [image_directory+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
    return image_list

def create_graph():
    with gfile.FastGFile(os.path.join(model_dir,
                    'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

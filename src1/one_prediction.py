import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import src1.feature_extraction as feature
from src1.image_resize import image_resize

"""
This script will take in a image path, and classify the image based on the
SVM that was trained on the features taken out of Inception V3
"""

def image_show(filepath):
    """
    Helper function to show the image being classified
    """
    img = mpimg.imread(filepath)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def predict_one(image_path):
    """
    Function takes in a image_path, and classifies what trick is being done.
    Args:
        Image path, does not need to be a string type

    Returns:
        Predicted class, and Probability of prediction. Also the image being
        pushed into the function
    """
    """
    The re-size will give not the image_path, so we need to save the resized image_path
    to a file location and use that new file location to be added into the feature vector object
    """

    re_size = image_resize([image_path])
    predicted_image_path = 'project_photos/one_prediction/predicted.jpg'
    feature_vector = feature.extraction([predicted_image_path])
    model_path = 'pickle_files/svm_model.pkl'
    with open(model_path, 'rb') as s:
        model = pickle.load(s)

    prediction_class = model.predict(feature_vector)
    prediction = model.predict_proba(feature_vector)

    print('\nYour prediction...')
    if prediction_class == 'kickflip'
        print(prediction_class, prediction[0])
        image_show(image_path)
    else:
        print(prediction_class, prediction[1])
        image_show(image_path)

# Photo paths that I have tested on.

# ../Capstone_photo/random_test_pictures/ollie_test_1.jpg
# ../Capstone_photo/random_test_pictures/grahamrussell_ollie1.jpg
# ../Capstone_photo/random_test_pictures/test_ollie_picture.jpg "Use for testing" False classification

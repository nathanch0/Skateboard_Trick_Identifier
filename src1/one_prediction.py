import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import feature_extraction as feature
from image_resize import image_resize as resize

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
    feature_vector = feature.extraction(resize([image_path]))
    model_path = '../pickle_files/svm_model.pkl'
    with open(model_path, 'rb') as s:
        model = pickle.load(s)

    prediction_class = model.predict(feature_vector)
    prediction = model.predict_proba(feature_vector)

    print('\nYour prediction...')
    print(prediction_class, prediction)
    image_show(image_path)

# Photo paths that I have tested on.

# ../Capstone_photo/random_test_pictures/ollie_test_1.jpg
# ../Capstone_photo/random_test_pictures/grahamrussell_ollie1.jpg
# ../Capstone_photo/random_test_pictures/test_ollie_picture.jpg "Use for testing" False classification

if __name__ == '__main__':
    image_path = input('Please input a file path for the image: ')
    predict_one(str(image_path))

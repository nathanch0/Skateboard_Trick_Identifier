import pickle
import feature_extraction as feature

"""
This script will take in a image path, and classify the image based on the
SVM that was trained on the features taken out of Inception V3
"""


def predict_one(image_path):
    feature_vector = feature.extraction([image_path])
    model = pickle.load('pickle_files/svm.pkl')

    prediction_class = model.predict(feature_vector)
    prediction = model.predict_proba(feature_vector)

    print('Your prediction...')
    return prediction_class, prediction




if __name__ == '__main__':
    image_path = raw_input('Please input a file path for the image: ')
    predict_one(image_path)

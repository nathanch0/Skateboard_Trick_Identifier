import os
import pickle

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

def train_svm_classifer(features, labels):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    Args:
        features: array of input features
        labels: array of labels associated with the input features
    """
    # Our test set will be 20% of the whole data set.
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    param = [{"kernel": ["linear"],
                "C": [1, 10, 100, 1000]},
             {"kernel": ["rbf"],
                "C": [1, 10, 100, 1000],
                "gamma": [1e-2, 1e-3, 1e-4, 1e-5]}]

    # Probability is requested as True
    svm = SVC(probability=True)

    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    SVM = GridSearchCV(svm, param,
            cv=10, n_jobs=4, verbose=3)

    # This will save the trained model in a pickle file to be used later
    model = SVM.fit(X_train, y_train)
    model_path = 'pickle_files/svm_model.pkl'
    with open(model_path,'wb') as f:
        pickle.dump(model,f)

    y_predict = model.predict(X_test)

    print("\nThe Best Parameters:")
    print(SVM.best_params_)

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

    # question = raw_input('Do you want to see how the model did? (yes/no):')
    # if question == 'yes':
    #
    #     labels=sorted(list(set(labels)))
    #     print("\nConfusion matrix:")
    #     print("Labels: {0}\n".format(",".join(labels)))
    #     print(confusion_matrix(y_test, y_predict, labels=labels))

# if __name__ == '__main__':
#     with open('pickle_files/features.pkl', 'rb') as f:
#         features = pickle.load(f)
#     with open('pickle_files/labels.pkl', 'rb') as l:
#         labels = pickle.load(l)
#     train_svm_classifer(features,labels)

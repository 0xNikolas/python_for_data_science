# Libraries
import numpy as np

# Classifiers
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Accuracy score
from sklearn.metrics import accuracy_score

# Training data [height, weight, shoe_size]
# Input
X = np.matrix([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
               [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])
# Labels
Y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male'])

# Create 3 more classifiers for the challenge
# Classifier comparison: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
classifiers = [tree.DecisionTreeClassifier(), MLPClassifier(alpha=1), SVC(C=0.025), KNeighborsClassifier()]

# Fit the data to each classifier
for classifier in classifiers:
    classifier.fit(X, Y)

# Make all predictions
Y_predictions = [clf.predict(X) for clf in classifiers]

# Compute accuracy score for each prediction on training data
accuracy_scores = [accuracy_score(Y, Y_pred) for Y_pred in Y_predictions]

# Extract best score index to access classifiers list with same index
best_score_idx = np.argmax(accuracy_scores)

# Voilla
print("The best score is {:.2f} for classifier {}.".format(accuracy_scores[best_score_idx],
                                                           type(classifiers[best_score_idx]).__name__))

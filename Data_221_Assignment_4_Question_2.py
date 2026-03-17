from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Data_221_Assignment_4_Question_1 import feature_matrix_X, labels_y

x_train, x_test, y_train, y_test = train_test_split(
    feature_matrix_X, labels_y, test_size=0.2, random_state=42
)

#Creating and training the decision tree
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(x_train, y_train)


def main():
    # Reporting accuracy for training set
    predicted_training = decision_tree_classifier.predict(x_train)
    print("Accuracy on training set:", accuracy_score(predicted_training, y_train))

    # Reporting accuracy for testing set
    predicted_testing = decision_tree_classifier.predict(x_test)
    print("Accuracy on testing set:", accuracy_score(predicted_testing, y_test))


if __name__ == "__main__":
    main()

#In the context of decision trees, entropy measures the goodness of a partition, with lower entropy
# values for certain partitions meaning that that partition does a very good job at splitting multiple
# classes apart, and higher values being unfavourable as they are not as informative

#I would say that the results given suggest decent generalisation, as the accuracy managed to stay over
# 90% for the testing set, but I do think it could be better when it comes to this; getting above 96%
# would suggest the model has done a good job at not overfitting to the training set
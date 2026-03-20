from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Data_221_Assignment_4_Question_2 import x_train, x_test, y_train, y_test
from Data_221_Assignment_4_Question_1 import breast_cancer_data
import pandas as pd

constrained_decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
constrained_decision_tree.fit(x_train, y_train)

def main():
    predicted_training = constrained_decision_tree.predict(x_train)
    print("Accuracy on training set:", accuracy_score(y_train, predicted_training))

    predicted_testing = constrained_decision_tree.predict(x_test)
    print("Accuracy on testing set:", accuracy_score(y_test, predicted_testing))

    print()

    #Saving the features and their importances into a pandas DataFrame
    features_and_importances = pd.DataFrame({
        #getting the feature names from the dataset
        "Feature": breast_cancer_data.feature_names,

        #getting the value of importance from the decision tree
        "importance": constrained_decision_tree.feature_importances_
    })

    #Sort the values numerically by importance in descending order, and take the first 5
    print(
        "The 5 most important features from the decision tree:\n",
        features_and_importances.sort_values("importance", ascending=False).head(5)
    )

if __name__ == "__main__":
    main()

#Controlling a model's complexity is essential to avoid overfitting, as if a model is not controlled,
# then the model will start to effectively memorise the training data and will not be able to adapt
# to rather unforeseen data.
#In the case of a decision tree, not controlling the maximum depth of a model will increase the
# complexity, which may be a problem as decision nodes may be overly specific to the training data

#Features with a higher importance will be the ones that are looked first when classifying whether
# someone is malignant or benign, as they are the most divisive ones, hence their high importance
#Extracting the feature importance from a decision tree can tell us which features contribute the
# most towards whether someone is of one classification or another
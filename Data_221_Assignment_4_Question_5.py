from sklearn.metrics import confusion_matrix
from Data_221_Assignment_4_Question_2 import y_test
import Data_221_Assignment_4_Question_3 as decision_tree
from Data_221_Assignment_4_Question_4 import get_y_pred

#Calculating the decision tree matrix for predicted testing
decision_tree_confusion_matrix = confusion_matrix(decision_tree.predicted_testing, y_test)

#Using the get_y_pred function from Question 4 to create the confusion matrix
#Note here that .astype(int) here is converting any predicted classification greater than 0.5
# to 1, as confusion_matrix() only works with binary classifications (1 or not 1) and any predicted
# classification that is greater than 0.5 is effectively 1.
#.astype() is from numpy and pandas
neural_network_confusion_matrix = confusion_matrix(
    (get_y_pred() >= 0.5).astype(int), y_test
)

#printing each model's respective confusion matrix
print("Decision tree confusion matrix:\n", decision_tree_confusion_matrix, "\n")
print("Neural network confusion matrix:\n", neural_network_confusion_matrix)


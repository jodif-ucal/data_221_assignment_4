from sklearn.metrics import confusion_matrix
from Data_221_Assignment_4_Question_2 import y_test
import Data_221_Assignment_4_Question_3 as decision_tree
from Data_221_Assignment_4_Question_4 import get_y_pred

#Calculating the decision tree matrix for predicted testing
decision_tree_confusion_matrix = confusion_matrix(y_test, decision_tree.predicted_testing, labels=[0,1])

#Using the get_y_pred function from Question 4 to create the confusion matrix
#Note here that .astype(int) here is converting any predicted classification greater than 0.5
# to 1, as confusion_matrix() only works with binary classifications (1 or not 1) and any predicted
# classification that is greater than 0.5 is effectively 1.
#.astype() is from numpy and pandas
neural_network_confusion_matrix = confusion_matrix(
    y_test, (get_y_pred() >= 0.5).astype(int), labels=[0,1]
)

#printing each model's respective confusion matrix
print("Decision tree confusion matrix:\n", decision_tree_confusion_matrix, "\n")
print("Neural network confusion matrix:\n", neural_network_confusion_matrix)

#For a task like this, I would prefer a Decision tree. This is because, for a problem like breast
# cancer, a decision tree will be able to capture hidden patterns between someone who is malignant
# and someone who is benign. A neural network is not designed to figure out what features are more
# important than others, however, as it takes all of them into account.
#We can see this reflected in each model's confusion matrix, as the decision tree managed to flag
# two more data points as true positives compared to the neural network, minimising the false
# negatives flagged, which is very important in the context of breast cancer.

#Advantage of decision trees: Able to identify the key features that goes into a certain classification,
# which can be useful outside a machine learning context
#Disadvantage of decision trees: Tend to overfitting, which will require careful management of them,
# which is why there are many hyperparameters to do so

#Advantage of neural networks: Versatile model that can work with many types of problems, including
# both classification (binary or multi-classification) and regression problems, and even image
# recognition
#Disadvantage of neural networks: Difficult to understand as we won't know why a model chooses certain
# weights and biases for each node, making it black box-like
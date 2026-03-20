from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.random import set_seed
from Data_221_Assignment_4_Question_2 import x_train, x_test, y_train, y_test
import numpy as np

#getting the mean and the standard deviation for standardisation of the features for testing and training
mean_test = np.mean(x_train)
mean_train = np.mean(x_test)
std_test = np.std(x_train)
std_train = np.std(x_test)

#Standardising both the testing and training data
x_train = (x_train - mean_train) / std_train
x_test = (x_test - mean_test) / std_test

#Setting a random seed so that the model's performance will remain consistent
set_seed(1)

#Setting up the neural network
neural_network_model = Sequential()

#Adding an input layer with 30 nodes
neural_network_model.add(InputLayer(input_shape=(30, )))

#Adding a hidden layer with 20 nodes
neural_network_model.add(Dense(20))

#Adding an output layer with 1 node, which will use a sigmoid curve to predict whether someone is
# malignant or benign
neural_network_model.add(Dense(1, activation="sigmoid"))

#Compiling the model
#Loss is binary crossentropy as we are only predicting whether someone is malignant or not
#Metrics is accuracy as we will test the model on the training and testing data later
neural_network_model.compile(loss="binary_crossentropy", metrics=["accuracy"])

#Training the model with 15 epochs (repetitions)
neural_network_model.fit(x_train, y_train, epochs=15)


def main():
    # Predicting labels for the testing and training data, and reporting the model's accuracy
    training_accuracy = neural_network_model.evaluate(x_train, y_train)
    testing_accuracy = neural_network_model.evaluate(x_test, y_test)

    # Printing the model's accuracy on the training and testing set
    # indexing by 1 as the loss of the model was also returned in a list
    print("Training accuracy:", training_accuracy[1])
    print("Testing accuracy:", testing_accuracy[1])

#This is for question 5
def get_y_pred():
    y_pred = neural_network_model.predict(x_test)
    return y_pred

if __name__ == "__main__":
    main()

#Feature scaling is important with neural networks because some features will have a larger variation
# between their data points than others, with the issue being that this larger variance will affect
# some of the weights and biases of each neuron, which every other feature will be affected by
#This could lead to the model being biased to features with high variance, so feature scaling
# reduces this by making the variance within each feature constant

#Epochs represent the number of times a model will repeatedly train itself on the training data
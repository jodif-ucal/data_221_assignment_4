import matplotlib.pyplot as plt
from Data_221_Assignment_4_Question_6 import cnn_model, X_test, y_test
from sklearn.metrics import confusion_matrix
import numpy as np

#Making the confusion matrix
#np.argmax() will take the probabilities given by the model and chose the label with the highest
# probability as the respective label for each range of probabilities given
predicted_labels = np.argmax(cnn_model.predict(X_test), axis=1)
matrix = confusion_matrix(y_test, predicted_labels, labels=[0,1])

#Printing the confusion matrix
print("Confusion matrix:\n", matrix, "\n")

count = 0
indexes = 0

#The index of the first 3 misclassified images will go here
index_of_misclassified_images = []

#Finding the first 3 misclassified images
while count < 3:
    if predicted_labels[indexes] != y_test[indexes]:
        index_of_misclassified_images.append(indexes)
        count += 1

    indexes += 1

#Displaaying the misclassified images
for index in index_of_misclassified_images:
    plt.gray()
    plt.matshow(X_test[index])
    plt.title(f"True label: {y_test[index]}; predicted label: {predicted_labels[index]}")
    plt.show()

print("The displayed images can be found in Question-7-images")

#One thing all the misclassified labels have in common is some sort of semicircle-like pattern present,
# with it being at the top of the object in images 1 and 3, and being further down in image 2

#One possible improvement that can be done to this model is by giving the model a larger stride during
# training, as it should be able to effectively capture the semicircle pattern found in the 3 misclassified
# images better.
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Normalising the pixels to be between [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

#Reshaping the image so that the CNN will be able to understand it
X_train = X_train[..., None]
X_test = X_test[..., None]

#Buidling the model
cnn_model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPool2D()
])
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(128, activation="relu"))
cnn_model.add(layers.Dropout(0.3)) #randomly drop 30% of nodes during training
cnn_model.add(layers.Dense(10, activation="softmax")) #Output layer
cnn_model.summary()

#Compiling the model
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Training the model on the training data
cnn_model.fit(
    X_train, y_train, validation_split=0.1, epochs=15, batch_size=64
)

#Getting and printing the test accuracy
test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

#CNNs are preferred over fully connected networks for image data, and this is because in oder to use
# the connected networks, the image will have to be flattened into a single long vector, meaning
# the model will not know which pixels are next to which and there will be a large number of neurons
# in the input layer alone, which will make the regular models slower and at risk of overfitting on
# any training data it sees.
#CNNs in contrast are able to sense patterns within an image and can combine these patterns into larger
# patterns and classify what is what accordingly using these patterns.

#The convolutional layer in this task is identifying some of the patterns that are present in fashion
# clothing, such as the handles of a bag and its general shape, before eventually building up to a larger,
# more complex shape, such as the bag itself.
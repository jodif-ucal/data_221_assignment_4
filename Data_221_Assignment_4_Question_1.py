from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()

feature_matrix = breast_cancer_data.data
labels = breast_cancer_data.target

def main():
    from numpy import count_nonzero

    print("Shape of x:", feature_matrix.shape) #569 rows, 30 columns
    print("Shape of y:", labels.shape) #569 rows, 1 column

    #numpy.count_nonzero takes a numpy array as input as well as the number to search for and
    # returns the number of times the number appears in the array
    #Note: Benign is classed as 0 and malignant as class 1
    print("Number of samples belonging to benign class:", count_nonzero(labels == 0))
    print("Number of samples belonging to malignant class:", count_nonzero(labels == 1))

#If this script is directly ran, the main function will run
#This way, we can import the data from here to other files without the print statements running again
if __name__ == "__main__":
    main()

#I would say the dataset is imbalanced as only 37% of samples belong to the benign class whilst
# 62% of samples belong to the malignant class

#Class balance is important to consider for classification models as such models need to have reasonable
# exposure to all classifications in order to not make decisions that are biased to or against certain
# classifications

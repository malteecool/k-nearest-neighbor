import functools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import *
import matplotlib.pyplot as plt
from sklearn import metrics


def train_knn(x_train, y_train, k):
    """
    Given training data (input and output), train a k-NN classifier.

    Input:    x/y_train - Two arrays of equal length, one with input data and 
              one with the correct labels. 
              k - number of neighbors considered when training the classifier.
    Returns:  The trained classifier
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    return knn


def evaluate_knn(knn, x_train, y_train, x_test, y_test):
    """
    Given a trained classifier, its training data, and test data, calculate
    the accuracy on the training and test sets.
    
    Input:    knn - A trained k-nn classifier
              x/y_train - Training data
              x/y_test  - Test data
    
    Returns:  A tuple (train_acc, test_acc) with the resulting accuracies,
              obtained when using the classifier on the given data.
    """
    train_score = knn.score(x_train, y_train)
    test_score = knn.score(x_test, y_test)
    return (train_score, test_score)


def load_dataset(name, features, test_size):
    """
    Loads the iris or breast cancer datasets with the given features and 
    train/test ratio.
    
    Input:    name - Either "iris" or "breastcancer"
              features - An array with the indicies of the features to load
              test_size - How large part of the dataset to be used as test data.
                          0.33 would give a test set 33% of the total size.
    Returns:  Arrays x_train, x_test, y_train, y_test that correspond to the
              training/test sets.
    """
    # Load the dataset
    if name == "iris":
        dataset = load_iris()
    elif name == "breastcancer":
        dataset = load_breast_cancer()

    print('You are using the features:')
    for x in features:
        print(x, "-", dataset.feature_names[x])

    X = dataset.data[:, features]
    Y = dataset.target

    # Split the dataset into a training and a test set
    # TODO choose a number as a seed (just to initialize the random number
    # generator.
    return train_test_split(X, Y, test_size=test_size, random_state=628875008)

# Iris ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
# features = [2, 3]
features = [1,2]


# Breast Cancer ['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',...
# features = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] = 1.0729742558775897
# features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #= 0.8687803652203054
# features = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] = 1.0708521807114535
# features = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] = 0.479142234880215
# features = [0, 2, 3, 10, 12, 13, 20, 22, 23, 9]

k_max = 30

x_train, x_test, y_train, y_test = load_dataset('iris', features, 0.33)

train_scores = []
test_scores = []

# TODO

best_acc = [0,0]
test = 0
for k in range(1,k_max):
    knn = train_knn(x_train, y_train, k)
    pred_y = knn.predict(x_test)
    acc = metrics.accuracy_score(y_test, pred_y)
    print("Accuracy of model is: ", acc)
    if(acc >= best_acc[1]):
        best_acc[1] = acc
        best_acc[0] = k

    
    train, test = evaluate_knn(knn, x_train, y_train, x_test, y_test)
    
    train_scores.append(train)
    test_scores.append(test)

print("Best Acc: ", best_acc[1], " At k: ", best_acc[0])

# Construct plot
plt.title('KNN results')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')

# Create x-axis
xaxis = [x for x in range(1,k_max)]
print(xaxis)

# Plot the test and training scores with labels
plt.plot(xaxis, train_scores, label='Training score')
plt.plot(xaxis, test_scores, label='Test score')

# Show the figure
plt.legend()
plt.show()

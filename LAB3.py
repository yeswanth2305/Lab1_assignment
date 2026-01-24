import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski

def dot(a, b):
    total = 0
    for i in range(len(a)):
        total = total + a[i] * b[i]
    return total



def length(a):
    total = 0
    for value in a:
        total = total + value * value
    return math.sqrt(total)



def avg(data):
    total = 0
    for value in data:
        total = total + value
    return total / len(data)


def var(data):
    m = avg(data)
    total = 0
    for value in data:
        total = total + (value - m) * (value - m)
    return total / len(data)

def center(class_data):
    return np.mean(class_data, axis=0)

def class_distance(c1, c2):
    return length(c1 - c2)

def mink_dist(a, b, p):
    total = 0
    for i in range(len(a)):
        total = total + abs(a[i] - b[i]) ** p
    return total ** (1 / p)

def split_data(features, labels):
    return train_test_split(features, labels, test_size=0.3, random_state=1)

def train_knn(train_features, train_labels, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_features, train_labels)
    return model

def accuracy(model, test_features, test_labels):
    return model.score(test_features, test_labels)

def predict(model, test_features):
    return model.predict(test_features)


def my_knn(train_features, train_labels, test_point, k):
    distance_list = []

    for i in range(len(train_features)):
        d = length(train_features[i] - test_point)
        distance_list.append((d, train_labels[i]))

    distance_list.sort()
    nearest = distance_list[:k]

    count_0 = 0
    count_1 = 0

    for item in nearest:
        if item[1] == 0:
            count_0 += 1
        else:
            count_1 += 1

    if count_0 > count_1:
        return 0
    else:
        return 1


def confusion(actual, predicted):
    TP = TN = FP = FN = 0

    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            TP += 1
        elif actual[i] == 0 and predicted[i] == 0:
            TN += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FP += 1
        else:
            FN += 1

    return TP, TN, FP, FN

def matrix_method(train_features, train_labels):
    ones = np.ones((len(train_features), 1))
    X = np.hstack((ones, train_features))
    w = np.linalg.inv(X.T @ X) @ X.T @ train_labels
    return w

def main():
    features = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7],
        [8, 6]
    ])

    labels = np.array([0, 0, 0, 1, 1, 1])

    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print("Dot product:", dot(v1, v2))
    print("Vector length:", length(v1))

   
    class0 = features[labels == 0]
    class1 = features[labels == 1]
    print("Distance between classes:",
          class_distance(center(class0), center(class1)))

   
    print("Minkowski distance:",
          mink_dist(v1, v2, 2), minkowski(v1, v2, 2))

   
    train_features, test_features, train_labels, test_labels = split_data(features, labels)

    
    model = train_knn(train_features, train_labels, 3)
    predicted = predict(model, test_features)
    print("kNN Accuracy:", accuracy(model, test_features, test_labels))

   
    print("Own kNN prediction:",
          my_knn(train_features, train_labels, test_features[0], 3))

    
    max_k = len(train_features)
    for k in range(1, max_k + 1):
        model_k = train_knn(train_features, train_labels, k)
        print("k =", k, "Accuracy =", accuracy(model_k, test_features, test_labels))

    
    TP, TN, FP, FN = confusion(test_labels, predicted)
    print("TP TN FP FN:", TP, TN, FP, FN)

    print("Matrix method weights:",
          matrix_method(train_features, train_labels))


main()

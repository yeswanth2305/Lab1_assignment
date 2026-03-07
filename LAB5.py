import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def load_dataset(path):

    data = pd.read_csv(path)

    if 'name' in data.columns:
        data = data.drop(columns=['name'])

    return data



def split_features_target(data):

    y = data['status']

    X = data.drop(columns=['status'])

    return X, y



def train_linear_regression(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mape, r2



def perform_kmeans(X, k):

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(X)

    labels = kmeans.labels_

    centers = kmeans.cluster_centers_

    return kmeans, labels, centers


def evaluate_clustering(X, labels):

    sil = silhouette_score(X, labels)

    ch = calinski_harabasz_score(X, labels)

    db = davies_bouldin_score(X, labels)

    return sil, ch, db



def evaluate_k_values(X):

    k_values = list(range(2,10))

    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in k_values:

        kmeans = KMeans(n_clusters=k, random_state=42)

        kmeans.fit(X)

        labels = kmeans.labels_

        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    return k_values, sil_scores, ch_scores, db_scores


def elbow_method(X):

    distortions = []

    k_range = range(2,20)

    for k in k_range:

        kmeans = KMeans(n_clusters=k, random_state=42)

        kmeans.fit(X)

        distortions.append(kmeans.inertia_)

    return k_range, distortions


def main():

    data = load_dataset("parkinsons.csv")

    X, y = split_features_target(data)


    

    X_single = X[['MDVP:Fo(Hz)']]

    X_train, X_test, y_train, y_test = train_test_split(
        X_single, y, test_size=0.2, random_state=42
    )

    model = train_linear_regression(X_train, y_train)

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_pred)

    test_metrics = calculate_metrics(y_test, test_pred)

    print("A1 Regression (Single Feature)")
    print("Train Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)


   

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_linear_regression(X_train, y_train)

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_pred)

    test_metrics = calculate_metrics(y_test, test_pred)

    print("\nA3 Regression (Multiple Features)")
    print("Train Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)


    

    kmeans_model, labels, centers = perform_kmeans(X, 2)

    print("\nA4 KMeans Clustering")
    print("Cluster Centers:")
    print(centers)


    

    sil, ch, db = evaluate_clustering(X, labels)

    print("\nA5 Clustering Scores")
    print("Silhouette Score:", sil)
    print("Calinski-Harabasz Score:", ch)
    print("Davies-Bouldin Score:", db)


    

    k_vals, sil_scores, ch_scores, db_scores = evaluate_k_values(X)

    plt.plot(k_vals, sil_scores)

    plt.xlabel("k")

    plt.ylabel("Silhouette Score")

    plt.title("Silhouette Score vs k")

    plt.show()



    k_range, distortions = elbow_method(X)

    plt.plot(k_range, distortions)

    plt.xlabel("Number of Clusters (k)")

    plt.ylabel("Distortion")

    plt.title("Elbow Method")

    plt.show()


if __name__ == "__main__":
    main()

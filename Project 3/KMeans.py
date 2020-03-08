# ML Project 3: K-Means Clustering on Iris Dataset
# Name: Shagun Paul
# UTA ID: 1001557958
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self, x_pts, y_pts):
        self.data_x_point = x_pts
        self.data_y_point = y_pts

    # Function to plot graphs for K-Means and Elbow Test
    def plt_centers(self, center_inf, pred_table):
        color_map = {0: 'purple', 1: 'magenta', 2: 'grey', 3: 'red', 4: 'cyan', 5: 'blue', 6: 'yellow', 7: 'brown',
                     8: 'green', 9: 'pink'}
        for kval, cent_val in center_inf.items():
            x_points = pred_table[pred_table['closest_to'] == kval]['x_val']
            y_points = pred_table[pred_table['closest_to'] == kval]['y_val']
            plt.scatter(x_points, y_points, c=color_map[kval])
            plt.scatter(cent_val[0], cent_val[1], s=100, marker="X", c="black")
        plt.show()

    # Function to randomly initialize the value of centroid K from given data points
    def initailize_k(self, X_val, Y_val, k_val):
        print("The Initialized Centroid Values: ")
        print("------------------------------------")
        random_init = [(i, np.random.randint(1, 150)) for i in range(k_val)]
        centroid = {
            r_val[0]: [X_val[r_val[1]], Y_val[r_val[1]]]
            for r_val in random_init
        }
        print(centroid)
        return centroid
    print("------------------------------------------------------------------------------------------------------")
    # Function to evaluate Euclidean Distance between input data points and centroids
    def eval_euclidean_dist(self, x_value, y_value, centroid):
        table = pd.DataFrame()
        # closest_val = []
        # Evaluates Euclidean Distance for each X and Y inputs for corresponding Centroid
        for k, val in centroid.items():
            table["distance from centroid {}".format(k)] = (
                np.sqrt(
                    (x_value - val[0]) ** 2
                    + (y_value - val[1]) ** 2
                )
            )
        # Fetches values of columns from Dataset
        column_vals = list(table.columns.values)
        """
        Selects all the rows for which the euclidean distance has been calculated and gets the column value which has minimum distance.
        """
        table['closest_to'] = table.loc[:, column_vals].idxmin(axis=1)
        table['closest_to'] = table['closest_to'].apply(lambda cls: int(cls.lstrip('distance from centroid ')))
        # This block helps in adding X_value and Y_value to the Data Frame and returns it.
        table['x_val'] = x_value
        table['y_val'] = y_value
        return table

    # Function to update centroid using mean values from each cluster
    def centroid_update(self, distance_table, centroid):
        new_centroids = centroid
        for k_val, centers in new_centroids.items():
            # The centroid is updated based on mean values of X_val
            new_centroids[k_val][0] = np.mean(distance_table[distance_table['closest_to'] == k_val]['x_val'])
            # The centroid is updated based on mean values of Y_val
            new_centroids[k_val][1] = np.mean(distance_table[distance_table['closest_to'] == k_val]['y_val'])
        return new_centroids

    # function to find Sum Squared Error in Elbow Test
    def SSE(self, c_info, res_table):
        values = []
        for k_val, value in c_info.items():
            X_pt = res_table[res_table['closest_to'] == k_val]['x_val']
            Y_pt = res_table[res_table['closest_to'] == k_val]['y_val']
            values.append(np.sum(np.power((X_pt - value[0]), 2) + np.power((Y_pt - value[1]), 2)))
        final = np.sum(values)
        return final

    # Function to perform K-means algorithm
    def kmeans(self, kclusters=3, elbow_test=False):
        print("K-Means Clustering is being Executed")
        centroids = self.initailize_k(self.data_x_point, self.data_y_point, kclusters)
        while True:
            flag = True
            # Used deep copy to draw comparison between new and old centroids.
            old_centroid = copy.deepcopy(centroids)
            resultant = self.eval_euclidean_dist(self.data_x_point, self.data_y_point, centroids)
            centroids = self.centroid_update(resultant, centroids)

            if not elbow_test:
                print("Old Centroid:")
                print(old_centroid)
                print("New Centroid:")
                print(centroids)
                self.plt_centers(old_centroid, resultant)
                print("-----------------------------------------------------------------------------------------------")
            # Counter to check whether the centroid is being changed or not
            for kclusters in centroids.keys():
                if (old_centroid[kclusters] == centroids[kclusters]) == False:
                    flag = False
            if flag:
                break
        # End of K-Means Algorithm
        error_value = self.SSE(centroids, resultant)
        return (centroids, resultant, error_value)

if __name__ == '__main__':
    attributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # Choosing Size of Cluster
    K = 3
    # Setting Class Value to Numerical Values
    class_values = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    dataset = pd.read_csv("iris_dataset.csv", names=attributes, header=None)
    dataset = dataset.replace({'class': class_values})
    print("-----------------------------------------------------------------------------------------------------------")
    print(dataset)
    print("Success! The Dataset has been loaded")

    sepal_length = dataset['sepal-length'].values
    sepal_width = dataset['sepal-width'].values
    petal_length = dataset['petal-length'].values
    petal_width = dataset['petal-width'].values

    # Creating object for class K-Means
    KM = KMeans(petal_length, petal_width)
    print("-----------------------------------------------------------------------------------------------------------")
    print(" Using Elbow Test Method \n \n")
    kval = [1, 2, 3, 5, 6, 7, 9]
    error = []

    for i in kval:
        info = KM.kmeans(i, elbow_test=True)
        print(info[0])
        error.append(info[2])

    print("-----------------------------------------------------------------------------------------------------------")
    print("\n The Sum Squared Error is:", error)
    print("-----------------------------------------------------------------------------------------------------------")

    plt.title("Elbow Test", fontsize=18)
    plt.xlabel("K Value", fontsize=13)
    plt.ylabel("Sum Squared Error Value", fontsize=13)
    plt.plot(kval, error, marker='o')
    plt.legend(["Sum Squared Value in relation to k"])
    plt.show()

    # Based on the result of Elbow test
    center_info, predicted, sse_val = KM.kmeans(kclusters=K)
    print(predicted)

    # Predictions are added to a csv file
    predicted.to_csv("results.csv")
    print("-----------------------------------------------------------------------------------------------------------")
    print("Success! The results are saved into results.csv file")

    if K == 3:
        predict = np.sort(predicted['closest_to'])
        true_values = np.sort(dataset['class'])
        actual_values = predict - true_values
        count_non0s = np.count_nonzero(actual_values)
        count_0s = len(actual_values) - count_non0s
        accuracy = (count_0s / len(actual_values)) * 100
        print("-------------------------------------------------------------------------------------------------------")
        print("\n The accuracy achieved by using K-Means Clustering is: {}%".format(accuracy))
        print("-------------------------------------------------------------------------------------------------------")
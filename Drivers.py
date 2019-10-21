import matplotlib.pyplot as plt
import p3andas as pd
import numpy as np
from sklearn.cluster import KMeans

color_map = {}


def main():
    end_program = 'no'
    print()
    while end_program == 'no':
        k = input_k()
        drivers_matrix = input_data()
        kmeans = kmeans_alg(drivers_matrix, k)
        save_results_pic(drivers_matrix, kmeans, k)
        # Calculate error
        print("For k = {} squared error value is {}".format(k, round(error_func(drivers_matrix,
                                                                     kmeans.cluster_centers_,
                                                                     kmeans.labels_))))
        end_program = input('Do you want to end program? (Enter no to continue): ')


def input_k():
    # input number of clusters
    while True:
        try:
            k = int(input("Please specify k: "))
            global color_map
            for i in range(k+1):
                color_map[i] = np.random.random(3)
            break
        except ValueError:
            print("Entered value is not correct. Please, enter correct value")
    return k


def input_data():
    # creating data frame from input file
    drivers_data_frame = pd.read_csv('drivers_dataset.txt', '\t')

    # Get columns from drivers data frame
    distance_feature = drivers_data_frame['Distance_Feature'].values
    speeding_feature = drivers_data_frame['Speeding_Feature'].values

    # Create matrix 2x2 from drivers data
    drivers_matrix = np.matrix(list(zip(distance_feature, speeding_feature)))
    return drivers_matrix


def kmeans_alg(drivers_matrix, k):
    # Kmeans algorithm
    kmeans = KMeans(n_clusters=k).fit(drivers_matrix)
    return kmeans


def save_results_pic(drivers_matrix, kmeans, k):
    plt.scatter([drivers_matrix[:, 0]], [drivers_matrix[:, 1]], c=[color_map[l] for l in kmeans.labels_], s=5)
    plt.show()
    plt.savefig('clusters_k{}.png'.format(k))


def error_func(drivers_matrix, clusters_centers, lables):

    k = len(clusters_centers)
    n = len(drivers_matrix)
    total_error = 0

    for j in range(k):
        sum_error = 0
        for i in range(n):
            if lables[i] == j:
                sum_error += np.linalg.norm(drivers_matrix[i, :] - clusters_centers[j, :]) ** 2
        total_error += sum_error

    return total_error


main()





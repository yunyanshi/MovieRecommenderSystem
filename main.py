# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum

def read_single_data(data_path):
    # Use a breakpoint in the code line below to debug your script.
    file = open(data_path, 'r')
    lines = file.readlines()
    data = []
    for line in lines:
        ratings = line.split()
        ratings = [int(rating) for rating in ratings]
        data.append(ratings)
    return np.array(data)

empty_filtered_training_data = 0

class Algorithm(Enum):
    COS = 1
    Pearson = 2

def process_pearson(user_id, training_data, rated_mid, rated_rating, predicted, k, mean_ratings, user_mean_rating_training):
    user_mean_rating_training = np.atleast_2d(user_mean_rating_training).T
    training_data = np.hstack((training_data, user_mean_rating_training))
    mean_rating_of_cur_user = round(np.mean(rated_rating))
    results = []
    for predicted_mid in predicted:
        filtered_training_data = training_data[training_data[:, predicted_mid] > 0]
        # column 1000 is the mean value of this user.

        # columns = [predicted_mid] + rated_mid

        # Version 2
        columns = [predicted_mid] + rated_mid + [1000]

        filtered_training_data = filtered_training_data[:, columns]
        filtered_training_data = filtered_training_data[~np.all(filtered_training_data[:,1:-1] == 0, axis=1)]
        cos_result = []

        if len(filtered_training_data) == 0:
            average_rating = mean_ratings[predicted_mid]
            if average_rating <= 0:
                average_rating = round(np.mean(rated_rating))
            results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
            continue

        for data in filtered_training_data:
            vec_1, vec_2 = [], []
            # index 0 is the rating in the training data. last value is the mean value of this user in the training data.
            for idx in range(1, len(data) - 1):
            # for idx in range(1, len(data)):
                if data[idx] == 0:
                    continue
                vec_1.append(data[idx])
                vec_2.append(rated_rating[idx - 1])
            # Edge Case #1: both vec_1 and vec_2 are one dimension vec.
            # if len(vec_1) == 1 and abs(vec_1[0] - vec_2[0]) > 1:
            #     continue
            if len(vec_1) == 1:
                continue

            vec_1.append(data[-1])
            # vec_1_mean, vec_2_mean = np.mean(vec_1), np.mean(vec_2)
            vec_1_mean, vec_2_mean = vec_1[-1], np.mean(rated_rating)
            vec_1_delta, vec_2_delta = vec_1 - vec_1_mean, vec_2 - vec_2_mean

            vec_1_delta = vec_1_delta[:-1]
            vec_1_delta_all_zeros = not np.any(vec_1_delta)
            if vec_1_delta_all_zeros is True:
                # print("1")
                vec_1_delta[0] = 0.05
                continue
            vec_2_delta_all_zeros = not np.any(vec_2_delta)
            if vec_2_delta_all_zeros is True:
                # print("2")
                vec_2_delta[0] = 0.05
                continue

            cos = dot(vec_1_delta, vec_2_delta) / (norm(vec_1_delta) * norm(vec_2_delta))

            # print(vec_1_delta)
            # print(vec_2_delta)
            # print(cos)
            # print("\n")
            cos_result.append([data[0], abs(cos), cos, vec_1_mean])

        if len(cos_result) == 0:
            average_rating = mean_ratings[predicted_mid]
            if average_rating <= 0:
                average_rating = round(np.mean(rated_rating))
        else:
            cos_result = np.array(cos_result)
            # print(cos_result)
            cos_result = np.atleast_2d(cos_result)
            # print(cos_result)
            cos_result_sorted = cos_result[cos_result[:, 1].argsort()]
            # Find Neighbors
            # Version 1: Top K
            cos_result_sorted_k = cos_result_sorted[-k:,]

            # Version 2: larger than a threshold
            cos_result_sorted_k = cos_result_sorted_k[cos_result_sorted_k[:,1] >= 0.7]
            average_rating = 0
            if len(cos_result_sorted_k) > 0:
                # average_rating = round(np.mean(cos_result_sorted_k[:,0]))
                # if average_rating == 0:
                #     average_rating = mean_ratings[predicted_mid]
                rated_mean = np.mean(rated_rating)
                weighted_delta = np.sum(cos_result_sorted_k[:, 2] * (cos_result_sorted_k[:,0] - cos_result_sorted_k[:, 3]))
                weight_sum = np.sum(cos_result_sorted_k[:, 1])
                if weighted_delta <= 0.05:
                    average_rating = round(rated_mean)
                else:
                    average_rating = round(rated_mean + weighted_delta / weight_sum)

                average_rating = min(5, average_rating)
                average_rating = max(1, average_rating)
            if average_rating == 0:
                # average_rating = mean_ratings[predicted_mid]
                average_rating = mean_rating_of_cur_user
            # average_rating = calculate_rating(Algorithm.Pearson, cos_result_sorted_k, rated_rating, mean_ratings[predicted_mid])
        results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
    return results

def process_cos(user_id, training_data, rated_mid, rated_rating, predicted, k, mean_ratings):
    results = []
    dim = len(rated_mid)
    for predicted_mid in predicted:
        filtered_training_data = training_data[training_data[:, predicted_mid] > 0]
        columns = [predicted_mid] + rated_mid
        filtered_training_data = filtered_training_data[:, columns]
        filtered_training_data = filtered_training_data[~np.all(filtered_training_data[:,1:] == 0, axis=1)]
        cos_result = []

        # this_sim = np.dot(filtered_training_data, rated_rating) / norm(training_user) * norm(test_user)
        # rated_norm = norm(rated_rating)
        # sim = dot(filtered_training_data[:,1], rated_rating) / (norm(filtered_training_data[:, 1]) * rated_norm)
        # cos_result = list(np.reshape(sim, -1))

        if len(filtered_training_data) == 0:
            average_rating = mean_ratings[predicted_mid]
            if average_rating <= 0:
                average_rating = round(np.mean(rated_rating))
            results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
            continue

        # print(cos_result.shape)
        for data in filtered_training_data:
            vec_1, vec_2 = [], []
            for idx in range(1, len(data)):
                if data[idx] == 0:
                    continue
                vec_1.append(data[idx])
                vec_2.append(rated_rating[idx - 1])
            # Edge Case #1: both vec_1 and vec_2 are one dimension vec.

            # Version 1
            # if len(vec_1) == 1 and abs(vec_1[0] - vec_2[0]) > 1:
            #     continue

            # Version 2: performs worse on test5.txt
            # if len(vec_1) == 1:
            #     continue

            # Version 3: assign a hard coded similarity
            if len(vec_1) == 1:
                if vec_1[0] - vec_2[0] == 0:
                    sim = 0.9
                elif abs(vec_1[0] - vec_2[0]) == 0:
                    sim = 0.8
                else:
                    continue
            else:
                sim = dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))
            cos_result.append([data[0], sim])

        if len(cos_result) == 0:
            average_rating = mean_ratings[predicted_mid]
            if average_rating <= 0:
                average_rating = round(np.mean(rated_rating))
        else:
            cos_result = np.array(cos_result)
            cos_result = np.atleast_2d(cos_result)
            cos_result_sorted = cos_result[cos_result[:, 1].argsort()]
            cos_result_sorted_k = cos_result_sorted[-k:,]
            # Wrong answer: mean
            # average_rating = round(np.mean(cos_result_sorted_k[:,0]))

            # Correct answer: weighted mean
            numerator = cos_result_sorted_k[:,0] * cos_result_sorted_k[:,1]
            denominator = np.sum(cos_result_sorted_k[:, 1])
            average_rating = round(np.sum(numerator) / denominator)
            if average_rating == 0:
                average_rating = mean_ratings[predicted_mid]
        results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
    return results

def predict_rating(training_data, test_data, neighbor_num, rated_num, algorithm):
    output_file = algorithm.name + '_result' + str(rated_num) + '.txt'
    cur_user_id = test_data[0][0]
    row = len(test_data)
    rated_mid, rated_rating, predicted = [], [], []
    results = []
    np.seterr(invalid='ignore')
    mean_ratings_tmp = np.true_divide(training_data.sum(0),(training_data!=0).sum(0))
    mean_ratings = []
    for x in mean_ratings_tmp:
        if np.isnan(x):
            mean_ratings.append(-1)
        else:
            mean_ratings.append(round(x))

    # Calculate mean ratings of each user in the training data
    user_mean_rating_training = np.true_divide(training_data.sum(1),(training_data!=0).sum(1))
    for r in tqdm(range(row), desc="Loading..."):
        user_id, movie_id, rating = test_data[r]
        # Convert movie_id to 0 based index
        movie_id -= 1
        if user_id == cur_user_id:
            if rating == 0:
                predicted.append(movie_id)
            else:
                rated_mid.append(movie_id)
                rated_rating.append(rating)
        if user_id != cur_user_id or r == row - 1:
            # Edge case: all ratings are the same
            all_ratings_are_equal = np.all(rated_rating == rated_rating[0])
            if all_ratings_are_equal:
                results_tmp = []
                for idx in range(len(predicted)):
                    results_tmp.append("{} {} {}".format(cur_user_id, predicted[idx] + 1, rated_rating[0]))
                results = results + results_tmp
            else:
                if algorithm is Algorithm.COS:
                    results = results + process_cos(cur_user_id, training_data, rated_mid, rated_rating, predicted, neighbor_num, mean_ratings)
                if algorithm is Algorithm.Pearson:
                    # print([cur_user_id] + rated_mid)
                    # print(rated_rating)
                    results = results + process_pearson(cur_user_id, training_data, rated_mid, rated_rating, predicted, neighbor_num, mean_ratings, user_mean_rating_training)
            cur_user_id = user_id
            rated_mid, rated_rating, predicted = [movie_id], [rating], []
            r -= 1
            # break
    with open(output_file, "a") as myfile:
        myfile.write("\n".join(results))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_data = read_single_data('train.txt')

    neighbor_num = 10
    algorithm = Algorithm.Pearson
    test5_data = read_single_data('test5.txt')
    predict_rating(training_data, test5_data, neighbor_num, 5, algorithm)

    test10_data = read_single_data('test10.txt')
    predict_rating(training_data, test10_data, neighbor_num, 10, algorithm)

    test20_data = read_single_data('test20.txt')
    predict_rating(training_data, test20_data, neighbor_num, 20, algorithm)

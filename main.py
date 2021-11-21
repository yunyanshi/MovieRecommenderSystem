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
import matplotlib.pyplot as plt
import util

empty_filtered_training_data = 0

class Algorithm(Enum):
    COS = 1
    Pearson = 2
    ItemBased = 3
    Custom = 4

def process_pearson(user_id, training_data, rated_mid, rated_rating, predicted, k, mean_ratings, user_ratings_mean):
    user_ratings_mean = np.atleast_2d(user_ratings_mean).T
    training_data = np.hstack((training_data, user_ratings_mean))
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

def process_custom(user_id, mean_ratings, std_ratings, rated_rating, predicted):
    results = []
    for predicted_mid in predicted:
        if std_ratings[predicted_mid] <= 1.0:
            rating = round(mean_ratings[predicted_mid])
        else:
            rating = round(np.mean(rated_rating))
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results

# This method returns a two dimensional array similarity. similarity[i][j] represents the similarity
# of movie i and movie j.
def calculate_movid_similarity(training_data):
    user_mean_rating_training = np.true_divide(training_data.sum(1),(training_data!=0).sum(1))
    user_mean_rating_training = user_mean_rating_training.reshape(user_mean_rating_training.shape[0], 1)
    adjusted_training_data = training_data
    row, column = training_data.shape
    similarity = np.zeros((column, column))
    count = 0
    for i in tqdm(range(column)):
        training_col_i = training_data[:, i]
        for j in range(column):
            training_col_j = training_data[:, j]
            v1, v2 = [], []
            for r in range(row):
                if training_col_i[r] == 0 or training_col_j[r] == 0:
                    continue
                v1.append(training_col_i[r] - user_mean_rating_training[r])
                v2.append(training_col_j[r] - user_mean_rating_training[r])
            if len(v1) <= 1:
                similarity[i, j] = None
            else:
                v1, v2 = np.hstack(v1), np.hstack(v2)
                value = np.dot(v1, v2.T) / (norm(v1) * norm(v2))
                if np.isnan(value):
                    value = None
                # similarity[i][j] = value
                # similarity[i][j] = (1.0 + value) / 2.0
                if value < -1.1 or value > 1.1:
                    print(value)
                    print("Wrong cos similarity value!!!!!!!!!!!!!!!!")
                theta = np.arccos(value) / 2.0
                similarity[i][j] = np.cos(theta)
                # if similarity[i][j]
    return similarity

def calculate_movid_similarity_v2(training_data):
    user_mean_rating_training = np.true_divide(training_data.sum(1),(training_data!=0).sum(1))
    user_mean_rating_training = user_mean_rating_training.reshape(user_mean_rating_training.shape[0], 1)
    adjusted_training_data = training_data - user_mean_rating_training
    row, column = training_data.shape
    similarity = np.zeros((column, column))
    for i in tqdm(range(column)):
        training_col_i = training_data[:, i]
        adjusted_training_data_col_i = adjusted_training_data[:, i]
        for j in range(column):
            training_col_j = training_data[:, j]
            adjusted_training_data_col_j = adjusted_training_data[:, j]

            product_of_i_j = training_col_i * training_col_j
            non_zero_indice = np.where(product_of_i_j != 0)[0]

            # non_zero_indice_i = np.where(training_col_i != 0)[0]
            # non_zero_indice_j = np.where(training_col_j != 0)[0]
            # non_zero_indice = np.intersect1d(non_zero_indice_i, non_zero_indice_j)
            if len(non_zero_indice) <= 1:
                similarity[i, j] = np.nan
            else:
                v1, v2 = adjusted_training_data_col_i[non_zero_indice], adjusted_training_data_col_j[non_zero_indice]
                # value = np.dot(v1, v2.T) / (norm(v1) * norm(v2))
                # if np.isnan(value):
                #     value = None
                # if value < -1.1 or value > 1.1:
                #     print(value)
                #     print("Wrong cos similarity value!!!!!!!!!!!!!!!!")
                # theta = np.arccos(value) / 2.0
                # similarity[i][j] = np.cos(theta)
                denominator = (norm(v1) * norm(v2))
                if denominator == 0:
                    similarity[i, j] = np.nan
                else:
                    value = np.dot(v1, v2.T) / denominator
                    if np.isnan(value):
                        similarity[i, j] = np.nan
                    else:
                        value = min(1.0, max(-1.0, value))
                        # theta = np.arccos(value) / 2.0
                        # similarity[i][j] = np.cos(theta)
                        similarity[i][j] = value
    return similarity

def process_item_based(user_id, similarity, rated_mid, rated_rating, predicted_mid, rated_num):
    results = []
    for predicted in predicted_mid:
        numerator, denominator = 0, 0
        for rated_idx in range(rated_num):
            rated_mid_tmp, rated_rating_tmp = rated_mid[rated_idx], rated_rating[rated_idx]
            sim = similarity[predicted][rated_mid_tmp]
            is_sim_nan = (sim == sim)
            if sim is None or np.isnan(sim) or is_sim_nan is False:
                continue
            # print(sim)
            # print(rated_rating_tmp)
            # print("===========")
            numerator += sim * rated_rating_tmp
            denominator += abs(sim)
        if denominator == 0:
            rating = round(np.mean(rated_rating))
        else:
            rating = round(numerator / denominator)
        if rating == 0:
            rating = round(np.mean(rated_rating))
            # print("numerator is 0.")
        results.append("{} {} {}".format(user_id, predicted + 1, rating))
    return results

def process_item_based_v2(user_id, similarity, rated_mid, rated_rating, predicted_mid, rated_num, neighbor_distribution):
    mean_rating_of_cur_user = np.mean(rated_rating)
    results = []
    for predicted in predicted_mid:
        numerator, denominator = 0, 0
        neighbor_count = 0
        for rated_idx in range(rated_num):
            rated_mid_tmp, rated_rating_tmp = rated_mid[rated_idx], rated_rating[rated_idx]
            sim = similarity[predicted][rated_mid_tmp]
            is_sim_nan = (sim == sim)
            if sim is None or np.isnan(sim) or is_sim_nan is False:
                continue
            neighbor_count += 1
            # print(sim)
            # print(rated_rating_tmp)
            # print("===========")
            numerator += sim * (rated_rating_tmp - mean_rating_of_cur_user)
            denominator += abs(sim)
        neighbor_distribution[neighbor_count] += 1
        if denominator == 0:
            rating = round(mean_rating_of_cur_user)
        else:
            rating = round(mean_rating_of_cur_user + numerator / denominator)
        if rating == 0:
            rating = round(mean_rating_of_cur_user)
        rating = min(5, max(1, rating))
        results.append("{} {} {}".format(user_id, predicted + 1, rating))
    return results

def predict_rating(training_data, test_data, neighbor_num, rated_num, algorithm, similarity):
    output_file = algorithm.name + '_result' + str(rated_num) + '.txt'

    np.seterr(invalid='ignore')
    movie_ratings_mean = util.get_mean_rating_of_each_movie(training_data)

    if algorithm is Algorithm.Custom:
        movie_ratings_std = util.get_rating_std_of_each_movie(training_data, movie_ratings_mean)
        # plt.hist(std_ratings, bins=50)
        # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
        # plt.show()
    if algorithm is Algorithm.Pearson:
        # Calculate mean ratings of each user in the training data
        user_ratings_mean = np.true_divide(training_data.sum(1),(training_data!=0).sum(1))

    neighbor_distribution = [0] * (rated_num + 1)
    results = []
    for r in tqdm(range(len(test_data)), desc="Loading..."):
        cur_user_test_data = test_data[r]
        cur_user_id, rated_mid, rated_rating, predicted = (
            cur_user_test_data['user_id'],
            cur_user_test_data['rated_mid'],
            cur_user_test_data['rating'],
            cur_user_test_data['predict_mid']
        )
        # Edge case: all ratings are the same
        all_ratings_are_equal = np.all(rated_rating == rated_rating[0])
        if all_ratings_are_equal:
            results_tmp = []
            for idx in range(len(predicted)):
                results_tmp.append("{} {} {}".format(cur_user_id, predicted[idx] + 1, rated_rating[0]))
            results = results + results_tmp
        else:
            if algorithm is Algorithm.COS:
                results = results + process_cos(cur_user_id, training_data, rated_mid, rated_rating, predicted, neighbor_num, movie_ratings_mean)
            if algorithm is Algorithm.Pearson:
                results = results + process_pearson(cur_user_id, training_data, rated_mid, rated_rating, predicted, neighbor_num, movie_ratings_mean, user_ratings_mean)
            if algorithm is Algorithm.Custom:
                results = results + process_custom(cur_user_id, movie_ratings_mean, movie_ratings_std, rated_rating, predicted)
            if algorithm is Algorithm.ItemBased:
                results_tmp = process_item_based_v2(cur_user_id, similarity, rated_mid, rated_rating, predicted, rated_num, neighbor_distribution)
                results = results + results_tmp
    with open(output_file, "a") as myfile:
        myfile.write("\n".join(results))
    print(neighbor_distribution)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_data = util.convert_training_data_to_2d_array('train.txt')

    neighbor_num = 10
    algorithm = Algorithm.Pearson

    similarity = None
    if algorithm is Algorithm.ItemBased:
        similarity = calculate_movid_similarity_v2(training_data)

    test5_data = util.convert_test_data_to_dict('test5.txt')
    predict_rating(training_data, test5_data, neighbor_num, 5, algorithm, similarity)

    # test10_data = util.convert_test_data_to_dict('test10.txt')
    # predict_rating(training_data, test10_data, neighbor_num, 10, algorithm, similarity)
    #
    # test20_data = util.convert_test_data_to_dict('test20.txt')
    # predict_rating(training_data, test20_data, neighbor_num, 20, algorithm, similarity)

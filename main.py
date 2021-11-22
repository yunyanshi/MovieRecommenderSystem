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
import pandas as pd

empty_filtered_training_data = 0

class Algorithm(Enum):
    COS = 1
    PEARSON = 2
    PEARSON_WITH_IUF = 3
    PEARSON_CASE_MOD = 4
    ITEM_BASED = 5
    CUSTOM = 6

def process_pearson(user_id, training_data, rated_mid, rated_rating, predicted, movie_ratings_mean,
                    movie_ratings_std, user_ratings_mean, algorithm, iuf, given_num):
    k = 20
    user_ratings_mean = np.atleast_2d(user_ratings_mean).T

    # column 1000 is the mean value of this user.
    training_data = np.hstack((training_data, user_ratings_mean))

    results = []
    for predicted_mid in predicted:
        # In training data, filter out all records whose predicted_mid rating is 0.
        filtered_training_data = training_data[training_data[:, predicted_mid] > 0]

        # column 1000 is the mean value of this user.
        columns = rated_mid + [predicted_mid, 1000]

        # pick the target columns from the training data
        filtered_training_data = filtered_training_data[:, columns]

        # Filter out records whose ratings are all 0.
        filtered_training_data = filtered_training_data[~np.all(filtered_training_data[:,:-2] == 0, axis=1)]

        if len(filtered_training_data) == 0:
            average_rating = get_default_movie_rating(predicted_mid, movie_ratings_mean, movie_ratings_std, rated_rating)
            results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
            continue

        cos_result = []
        for data in filtered_training_data:
            vec_1, vec_2 = [], []
            selected_mid = []
            for idx in range(len(data) - 2):
                if data[idx] == 0:
                    continue
                vec_1.append(data[idx])
                vec_2.append(rated_rating[idx])

                selected_mid.append(idx)
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
                vec_1_delta[0] = 0.05
                continue
            vec_2_delta_all_zeros = not np.any(vec_2_delta)
            if vec_2_delta_all_zeros is True:
                vec_2_delta[0] = 0.05
                continue

            cos = dot(vec_1_delta, vec_2_delta) / (norm(vec_1_delta) * norm(vec_2_delta))

            if algorithm is Algorithm.PEARSON_CASE_MOD:
                cos = cos * np.power(abs(cos), 1.5)

            iuf_sim = None
            if algorithm is Algorithm.PEARSON_WITH_IUF:
                selected_iuf = iuf[selected_mid]
                vec_1_delta_iuf, vec_2_delta_iuf = vec_1_delta * selected_iuf, vec_2_delta * selected_iuf
                iuf_sim = dot(vec_1_delta_iuf, vec_2_delta_iuf) / (norm(vec_1_delta_iuf) * norm(vec_2_delta_iuf))
                iuf_sim = abs(iuf_sim)

            cos_result.append([data[-2], abs(cos), cos, vec_1_mean, iuf_sim])

        if len(cos_result) == 0:
            average_rating = get_default_movie_rating(predicted_mid, movie_ratings_mean, movie_ratings_std, rated_rating)
        else:
            cos_result = np.array(cos_result)
            cos_result = np.atleast_2d(cos_result)

            # Find Neighbors
            # Version 1: Top K
            if algorithm in {Algorithm.PEARSON, Algorithm.PEARSON_CASE_MOD}:
                cos_result_sorted = cos_result[cos_result[:, 1].argsort()]
            elif algorithm is Algorithm.PEARSON_WITH_IUF:
                cos_result_sorted = cos_result[cos_result[:, 4].argsort()]
            cos_result_sorted_k = cos_result_sorted[-k:,]

            # print(cos_result_sorted_k)
            # print("\n")

            # Version 2: larger than a threshold
            # cos_result_sorted_k = cos_result[cos_result[:,1] >= 0.6]
            # print(len(cos_result_sorted_k))

            average_rating = 0
            if len(cos_result_sorted_k) > 0:
                # average_rating = round(np.mean(cos_result_sorted_k[:,0]))
                # if average_rating == 0:
                #     average_rating = mean_ratings[predicted_mid]
                rated_mean = np.mean(rated_rating)
                weighted_delta = np.sum(cos_result_sorted_k[:, 2] * (cos_result_sorted_k[:,0] - cos_result_sorted_k[:, 3]))
                weight_sum = np.sum(cos_result_sorted_k[:, 1])
                if weighted_delta <= 0.005:
                    average_rating = get_default_movie_rating(predicted_mid, movie_ratings_mean, movie_ratings_std, rated_rating)
                else:
                    average_rating = round(rated_mean + weighted_delta / weight_sum)

                average_rating = min(5, average_rating)
                average_rating = max(1, average_rating)
            if average_rating == 0:
                average_rating = get_default_movie_rating(predicted_mid, movie_ratings_mean, movie_ratings_std, rated_rating)
        results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
    return results


def get_k_value(given_num):
    return given_num

def process_cos(user_id, training_data, rated_mid, rated_rating, predicted, movie_ratings_mean, movie_ratings_std, given_num):
    k = get_k_value(given_num)

    results = []
    for predicted_mid in predicted:
        # remove the rows that don't have a rating for the movie with id of predicted_mid
        filtered_training_data = training_data[training_data[:, predicted_mid] > 0]

        # remove the columns that the active user doesn't have a rating for
        columns = [predicted_mid] + rated_mid
        filtered_training_data = filtered_training_data[:, columns]

        # remove rows that have a zero for all the movies the active user has rated
        filtered_training_data = filtered_training_data[~np.all(filtered_training_data[:,1:] == 0, axis=1)]

        cos_result = []
        if len(filtered_training_data) == 0:
            average_rating = get_default_movie_rating(predicted_mid, movie_ratings_mean, movie_ratings_std, rated_rating)
            results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
            continue

        for data in filtered_training_data:
            vec_1, vec_2 = [], []
            for idx in range(1, len(data)):
                if data[idx] == 0:
                    continue
                vec_1.append(data[idx])
                vec_2.append(rated_rating[idx - 1])

            if len(vec_1) <= 1:
                if vec_1[0] == vec_2[0]:
                    sim = 0.75
                else:
                    continue
            else:
                sim = dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))
            cos_result.append([data[0], sim])

        if len(cos_result) == 0:
            average_rating = get_default_movie_rating(predicted_mid, movie_ratings_mean, movie_ratings_std,
                                                      rated_rating)
        else:
            # print(len(cos_result))
            cos_result = np.array(cos_result)
            cos_result = np.atleast_2d(cos_result)
            cos_result_sorted = cos_result[cos_result[:, 1].argsort()]
            cos_result_sorted_k = cos_result_sorted[-k:,]

            # print(cos_result_sorted_k)
            # print("\n")

            # weighted average
            numerator = np.dot(cos_result_sorted_k[:,0], cos_result_sorted_k[:,1])
            denominator = np.sum(cos_result_sorted_k[:, 1])
            average_rating = round(np.sum(numerator) / denominator)

        results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
    return results

def get_default_movie_rating(mid, movie_ratings_mean, movie_ratings_std, user_ratings):
    if movie_ratings_std[mid] <= 1.0:
        return round(movie_ratings_mean[mid])
    else:
        return round(np.mean(user_ratings))

def process_custom(user_id, mean_ratings, std_ratings, rated_rating, predicted):
    results = []
    for predicted_mid in predicted:
        rating = get_default_movie_rating(predicted_mid, mean_ratings, std_ratings, rated_rating)
        # if std_ratings[predicted_mid] <= 1.0:
        #     rating = round(mean_ratings[predicted_mid])
        # else:
        #     rating = round(np.mean(rated_rating))
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results

def process_custom_v2(user_id, mean_ratings, std_ratings, rated_rating, rated_rating_std, predicted):
    mean_rating_of_active_user = np.mean(rated_rating)

    results = []
    for predicted_mid in predicted:
        movie_rating_std = std_ratings[predicted_mid]
        movie_rating = mean_ratings[predicted_mid]

        if movie_rating_std <= 0.5:
            rating = round(movie_rating)
        elif movie_rating_std < 1.25 and rated_rating_std < 1.25:
            rating = (1 / movie_rating_std * movie_rating + 1 / rated_rating_std * mean_rating_of_active_user) / (1 / movie_rating_std + 1 / rated_rating_std)
            rating = round(rating)
        elif movie_rating_std < rated_rating_std:
            rating = round(movie_rating)
        else:
            rating = round(mean_rating_of_active_user)
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results

def process_custom_v3(user_id, mean_ratings, std_ratings, rated_rating, rated_rating_std, predicted, given_num):
    # mean_rating_of_active_user = np.mean(rated_rating)
    #
    # results = []
    # for predicted_mid in predicted:
    #     movie_rating_std = std_ratings[predicted_mid]
    #     movie_rating = mean_ratings[predicted_mid]
    #
    #     if movie_rating_std <= 0.5:
    #         rating = round(movie_rating)
    #     elif given_num == 20 and rated_rating_std < 0.65:
    #         rating = round(mean_rating_of_active_user)
    #     elif given_num == 10 and rated_rating_std < 0.60:
    #         rating = round(mean_rating_of_active_user)
    #     elif given_num == 5 and rated_rating_std < 0.50:
    #         rating = round(mean_rating_of_active_user)
    #     elif movie_rating_std < 1.20 and rated_rating_std < 1.20:
    #         rating = (1 / movie_rating_std * movie_rating + 1 / rated_rating_std * mean_rating_of_active_user) / (1 / movie_rating_std + 1 / rated_rating_std)
    #         rating = round(rating)
    #     elif movie_rating_std < rated_rating_std:
    #         rating = round(movie_rating)
    #     else:
    #         rating = round(mean_rating_of_active_user)
    #     results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    # return results

    mean_rating_of_active_user = np.mean(rated_rating)

    results = []
    for predicted_mid in predicted:
        movie_rating_std = std_ratings[predicted_mid]
        movie_rating = mean_ratings[predicted_mid]

        if movie_rating_std <= 0.25:
            rating = round(movie_rating)
        elif movie_rating_std < 1.25 and rated_rating_std < 1.30:
            rating = (1 / movie_rating_std * movie_rating + 1 / rated_rating_std * mean_rating_of_active_user) / (1 / movie_rating_std + 1 / rated_rating_std)
            rating = round(rating)
        elif movie_rating_std < rated_rating_std:
            rating = round(movie_rating)
        else:
            rating = round(mean_rating_of_active_user)
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results


def process_custom_v4(user_id, mean_ratings, std_ratings, rated_rating, rated_rating_std, predicted, given_num):
    mean_rating_of_active_user = np.mean(rated_rating)

    results = []
    for predicted_mid in predicted:
        movie_rating_std = std_ratings[predicted_mid]
        movie_rating = mean_ratings[predicted_mid]

        if movie_rating_std <= 0.1:
            rating = round(movie_rating)
        elif movie_rating_std < 1.50 and rated_rating_std < 1.50:
            rating = (1 / movie_rating_std * movie_rating + 1 / rated_rating_std * mean_rating_of_active_user) / (1 / movie_rating_std + 1 / rated_rating_std)
            rating = round(rating)
        elif movie_rating_std < rated_rating_std:
            rating = round(movie_rating)
        else:
            rating = round(mean_rating_of_active_user)
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results

def process_custom_v5(user_id, mean_ratings, std_ratings, rated_rating, rated_rating_std, predicted, given_num):
    mean_rating_of_active_user = np.mean(rated_rating)

    results = []
    for predicted_mid in predicted:
        movie_rating_std = std_ratings[predicted_mid]
        movie_rating = mean_ratings[predicted_mid]

        if movie_rating_std <= 0.001:
            rating = round(movie_rating)
        elif movie_rating_std < 1.85 and rated_rating_std < 1.85:
            rating = (1 / movie_rating_std * movie_rating + 1 / rated_rating_std * mean_rating_of_active_user) / (1 / movie_rating_std + 1 / rated_rating_std)
            rating = round(rating)
        elif movie_rating_std < rated_rating_std:
            rating = round(movie_rating)
        else:
            rating = round(mean_rating_of_active_user)
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results

def process_custom_v6(user_id, mean_ratings, std_ratings, rated_rating, rated_rating_std, predicted, given_num):
    mean_rating_of_active_user = np.mean(rated_rating)

    results = []
    for predicted_mid in predicted:
        movie_rating_std = std_ratings[predicted_mid]
        movie_rating = mean_ratings[predicted_mid]

        if movie_rating_std <= 0.001:
            rating = round(movie_rating)
        else:
            rating = (1 / movie_rating_std * movie_rating + 1 / rated_rating_std * mean_rating_of_active_user) / (1 / movie_rating_std + 1 / rated_rating_std)

            if rating != rating:
                if movie_rating_std < rated_rating_std:
                    rating = round(movie_rating)
                else:
                    rating = round(mean_rating_of_active_user)
            else:
                rating = round(rating)
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results

def process_custom_v100(user_id, mean_ratings, std_ratings, rated_rating, rated_rating_std, predicted, model):
    # 0.77286
    mean_rating_of_active_user = np.mean(rated_rating)

    results = []
    for predicted_mid in predicted:
        movie_rating_std = std_ratings[predicted_mid]
        movie_rating = mean_ratings[predicted_mid]

        # std is nan
        if movie_rating_std != movie_rating_std:
            rating = round(mean_rating_of_active_user)
        else:
            model_input_data = {'predict_movie_mean' : [movie_rating], 'predict_movie_std' : [movie_rating_std],
                      'given_rating_mean' : [mean_rating_of_active_user], 'given_rating_std' : [rated_rating_std]}
            model_input_data = pd.DataFrame(model_input_data, columns=['predict_movie_mean', 'predict_movie_std', 'given_rating_mean', 'given_rating_std'])
            model_predict_rating = model.predict(model_input_data)
            rating = round(model_predict_rating[0])
            rating = min(5, max(1, rating))
        results.append("{} {} {}".format(user_id, predicted_mid + 1, rating))
    return results

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

def predict_rating(training_data, test_data, neighbor_num, given_num, algorithm, similarity, movie_ratings_mean, movie_ratings_std, user_ratings_mean, iuf, model):
    output_file = algorithm.name + '_result' + str(given_num) + '.txt'
    neighbor_distribution = [0] * (given_num + 1)
    results = []
    for r in tqdm(range(len(test_data)), desc="Loading..."):
        cur_user_test_data = test_data[r]
        cur_user_id, rated_mid, rating, rating_std, predict_mid = (
            cur_user_test_data['user_id'],
            cur_user_test_data['rated_mid'],
            cur_user_test_data['rating'],
            cur_user_test_data['std'],
            cur_user_test_data['predict_mid']
        )
        # Edge case: all ratings are the same
        all_ratings_are_equal = np.all(rating == rating[0])
        if all_ratings_are_equal:
            results_tmp = []
            for idx in range(len(predict_mid)):
                results_tmp.append("{} {} {}".format(cur_user_id, predict_mid[idx] + 1, rating[0]))
            results = results + results_tmp
        else:
            if algorithm is Algorithm.COS:
                results = results + process_cos(cur_user_id, training_data, rated_mid, rating, predict_mid,
                                                movie_ratings_mean, movie_ratings_std, given_num)
            if algorithm is Algorithm.PEARSON:
                results = results + process_pearson(cur_user_id, training_data, rated_mid, rating, predict_mid, movie_ratings_mean, movie_ratings_std, user_ratings_mean, algorithm, iuf, given_num)
            if algorithm is Algorithm.PEARSON_WITH_IUF:
                results = results + process_pearson(cur_user_id, training_data, rated_mid, rating, predict_mid,
                                                    movie_ratings_mean, movie_ratings_std,
                                                    user_ratings_mean, algorithm, iuf, given_num)
            if algorithm is Algorithm.PEARSON_CASE_MOD:
                results = results + process_pearson(cur_user_id, training_data, rated_mid, rating, predict_mid,
                                                    movie_ratings_mean, movie_ratings_std,
                                                    user_ratings_mean, algorithm, iuf, given_num)
            if algorithm is Algorithm.CUSTOM:
                # results = results + process_custom(cur_user_id, movie_ratings_mean, movie_ratings_std, rating, predict_mid)
                results = results + process_custom_v6(cur_user_id, movie_ratings_mean, movie_ratings_std, rating, rating_std,
                                                      predict_mid, given_num)
            if algorithm is Algorithm.ITEM_BASED:
                results = results + process_item_based_v2(cur_user_id, similarity, rated_mid, rating, predict_mid, given_num, neighbor_distribution)
    with open(output_file, "a") as myfile:
        myfile.write("\n".join(results))
    # print(neighbor_distribution)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.seterr(invalid='ignore')
    """
    convert train.txt to a 200*1000 matrix
    row represents user (uid: 1-200)
    column represents movie (mid: 1-1000)
    """
    training_data = util.convert_training_data_to_2d_array('train.txt')

    """
    k is the number of neighbors selected for each prediction
    """
    k = 20

    """
    algorithm specifies the algorithm that is being used for making predictions
    algorithm can be selected from:
    class Algorithm(Enum):
        COS = 1
        PEARSON = 2
        PEARSON_WITH_IUF = 3
        PEARSON_CASE_MOD = 4
        ITEM_BASED = 5
        CUSTOM = 6
        
    Please make a selection here for the algorithm you would like to test
    """
    algorithm = Algorithm.CUSTOM

    """
    If the algorithm is ITEM_BASED. a matrix that stores all the pairwise similarities
    between two movies will be created. 
    """
    pairwise_m_similarity = None
    if algorithm is Algorithm.ITEM_BASED:
        pairwise_m_similarity = util.calculate_movid_similarity_v2(training_data)

    iuf = None
    if algorithm is Algorithm.PEARSON_WITH_IUF:
        iuf = util.calculate_iuf(training_data)

    model_5, model_10, model_20 = None, None, None
    # if algorithm is Algorithm.CUSTOM:
    #     # Only run once
    #     # util.prepare_training_data(training_data, 5)
    #     # util.prepare_training_data(training_data, 10)
    #     model_5 = util.get_decision_tree_model(5)
    #     model_10 = util.get_decision_tree_model(10)
    #     model_20 = model_10

    movie_ratings_mean = util.get_mean_rating_of_each_movie(training_data)
    movie_ratings_std = util.get_rating_std_of_each_movie(training_data, movie_ratings_mean)
    user_ratings_mean = np.true_divide(training_data.sum(1), (training_data != 0).sum(1))

    # plt.hist(movie_ratings_std, bins=50)
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    # plt.show()

    test5_data = util.convert_test_data_to_dict('test5.txt')

    # active_user_ratings_std = []
    # for record in test5_data:
    #     active_user_ratings_std.append(record['std'])
    # plt.hist(active_user_ratings_std, bins=50)
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    # plt.show()

    predict_rating(training_data, test5_data, k, 5, algorithm, pairwise_m_similarity,
                   movie_ratings_mean, movie_ratings_std, user_ratings_mean, iuf, model_5)
    # #
    test10_data = util.convert_test_data_to_dict('test10.txt')
    # active_user_ratings_std = []
    # for record in test10_data:
    #     active_user_ratings_std.append(record['std'])
    # plt.hist(active_user_ratings_std, bins=50)
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    # plt.show()
    predict_rating(training_data, test10_data, k, 10, algorithm, pairwise_m_similarity,
                   movie_ratings_mean, movie_ratings_std, user_ratings_mean, iuf, model_10)
    #
    test20_data = util.convert_test_data_to_dict('test20.txt')
    # active_user_ratings_std = []
    # for record in test20_data:
    #     active_user_ratings_std.append(record['std'])
    # plt.hist(active_user_ratings_std, bins=50)
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    # plt.show()
    predict_rating(training_data, test20_data, k, 20, algorithm, pairwise_m_similarity,
                   movie_ratings_mean, movie_ratings_std, user_ratings_mean, iuf, model_20)

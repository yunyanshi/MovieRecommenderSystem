# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

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

def process(user_id, training_data, rated_mid, rated_rating, predicted, k, mean_ratings):
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

        # a = np.dot(rated_rating, filtered_training_data[:,1:].T)
        # tmp = a / (norm(rated_rating) * norm(filtered_training_data[:,1:].T))
        # cos_result = np.reshape(tmp, -1)
        # cos_result = np.vstack((filtered_training_data[:,1], cos_result)).T

        # print(cos_result.shape)
        for data in filtered_training_data:
            vec_1, vec_2 = [], []
            for idx in range(1, len(data)):
                if data[idx] == 0:
                    continue
                vec_1.append(data[idx])
                vec_2.append(rated_rating[idx - 1])
            # Edge Case #1: both vec_1 and vec_2 are one dimension vec.
            if len(vec_1) == 1 and abs(vec_1[0] - vec_2[0]) > 1:
                continue
            sim = dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))

            # sim = dot(data[1:], rated_rating) / (norm(data[1:]) * rated_norm)
            cos_result.append([data[0], sim])

        # print(filtered_training_data_3)
        # print(len(filtered_training_data_3))
        # print(cos_result)
        if len(cos_result) == 0:
            average_rating = mean_ratings[predicted_mid]
            if average_rating <= 0:
                average_rating = round(np.mean(rated_rating))
        else:
            cos_result = np.array(cos_result)
            cos_result = np.atleast_2d(cos_result)
            cos_result_sorted = cos_result[cos_result[:, 1].argsort()]
            # print(cos_result_sorted)
            cos_result_sorted_k = cos_result_sorted[-k:,]
            # print(cos_result_sorted_k)
            average_rating = round(np.mean(cos_result_sorted_k[:,0]))
            if average_rating == 0:
                # print(predicted_mid)
                average_rating = mean_ratings[predicted_mid]
        # print(average_rating)
        results.append("{} {} {}".format(user_id, predicted_mid + 1, average_rating))
    return results
        # with open("result5.txt", "a") as myfile:
        #     myfile.write("{} {} {}\n".format(user_id, predicted_mid, average_rating))

def predict_rating(training_data, test_data, k, output_file):
    cur_user_id = test_data[0][0]
    row = len(test_data)
    rated_mid, rated_rating, predicted = [], [], []
    results = []
    # tmp = training_data.sum(axis=0)
    # count = 0
    # for i in tmp:
    #     if i == 0:
    #         count += 1
    # print("======")
    # print(count)
    np.seterr(invalid='ignore')
    # mean_ratings_of_user = np.true_divide(training_data.sum(1),(training_data!=0).sum(1))
    mean_ratings_tmp = np.true_divide(training_data.sum(0),(training_data!=0).sum(0))
    mean_ratings = []
    for x in mean_ratings_tmp:
        if np.isnan(x):
            mean_ratings.append(-1)
        else:
            mean_ratings.append(round(x))
    # print(len(mean_ratings))
    # mean_ratings = [1 if np.isnan(x) is True else round(np.round(x)) for x in mean_ratings]
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
            results = results + process(cur_user_id, training_data, rated_mid, rated_rating, predicted, k, mean_ratings)
            cur_user_id = user_id
            rated_mid, rated_rating, predicted = [], [], []
            r -= 1
            # break
    with open(output_file, "a") as myfile:
        myfile.write("\n".join(results))

    # print(rated_mid)
    # print(rated_rating)
    # print(predicted)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_data = read_single_data('train.txt')

    test5_data = read_single_data('test5.txt')
    predict_rating(training_data, test5_data, 5, 'result5.txt')

    test10_data = read_single_data('test10.txt')
    predict_rating(training_data, test10_data, 5, 'result10.txt')

    test20_data = read_single_data('test20.txt')
    predict_rating(training_data, test20_data, 5, 'result20.txt')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

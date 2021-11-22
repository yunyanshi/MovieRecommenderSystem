
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

def convert_training_data_to_2d_array(txt_path):
    # Use a breakpoint in the code line below to debug your script.
    file = open(txt_path, 'r')
    lines = file.readlines()
    data = []
    for line in lines:
        ratings = line.split()
        ratings = [int(rating) for rating in ratings]
        data.append(ratings)
    return np.array(data)

def convert_test_data_to_dict(txt_path):
    results = []
    array = convert_training_data_to_2d_array(txt_path)
    row, col = array.shape

    rated_mid, rated_rating, predicted = [], [], []
    cur_user_id = array[0][0]

    for r in range(row):
        user_id, movie_id, rating = array[r]
        # Convert movie_id to 0 based index
        movie_id -= 1
        if user_id == cur_user_id:
            if rating == 0:
                predicted.append(movie_id)
            else:
                rated_mid.append(movie_id)
                rated_rating.append(rating)
        if user_id != cur_user_id or r == row - 1:
            # calculate std of cur active user.
            mean_tmp = np.mean(rated_rating)
            numerator, denominator = 0, 0
            for rating_tmp in rated_rating:
                numerator += np.power(rating_tmp - mean_tmp, 2)
            std = np.sqrt(numerator / len(rated_rating))

            results.append({'user_id' : cur_user_id, 'rated_mid' : rated_mid, 'rating' : rated_rating, 'predict_mid' : predicted, 'std': std})
            cur_user_id = user_id
            rated_mid, rated_rating, predicted = [movie_id], [rating], []
            r -= 1
    # print(results)
    return results

def get_mean_rating_of_each_movie(training_data):
    mean_ratings_tmp = np.true_divide(training_data.sum(0),(training_data!=0).sum(0))
    mean_ratings = []
    for x in mean_ratings_tmp:
        if np.isnan(x):
            mean_ratings.append(-1)
        else:
            mean_ratings.append(x)
    return mean_ratings

def get_rating_std_of_each_movie(training_data, mean_ratings):
    std_ratings = []
    for c in range(1000):
        if mean_ratings == -1:
            std_ratings.append(1000)
            continue
        tmp = training_data[:, c]
        tmp = tmp[tmp > 0]
        tmp = tmp - mean_ratings[c]
        N = len(tmp)
        numerator = tmp.dot(tmp.T)
        numerator = np.sum(numerator)
        std_ratings.append(np.sqrt(numerator / N))
    return std_ratings

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

def calculate_iuf(training_data):
    non_zero_count = np.count_nonzero(training_data, axis=0)
    result = np.log(np.divide(200, non_zero_count))
    for i in range(len(result)):
        if np.isinf(result[i]):
            result[i] = np.nan
    # print(result)
    # print(result.shape)
    return result

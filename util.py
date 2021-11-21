
import numpy as np
from tqdm import tqdm

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
            results.append({'user_id' : cur_user_id, 'rated_mid' : rated_mid, 'rating' : rated_rating, 'predict_mid' : predicted})
            cur_user_id = user_id
            rated_mid, rated_rating, predicted = [movie_id], [rating], []
            r -= 1
    print(results)
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
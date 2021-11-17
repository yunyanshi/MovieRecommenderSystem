def read_single_data(data_path):
    # Use a breakpoint in the code line below to debug your script.
    file = open(data_path, 'r')
    lines = file.readlines()
    data = []
    for line in lines:
        ratings = line.split()
        ratings = [int(rating) for rating in ratings]
        rating = ratings[2]
        if rating <= 0 or rating > 5:
            print(ratings)

read_single_data('Pearson_result5.txt')
read_single_data('Pearson_result10.txt')
read_single_data('Pearson_result20.txt')


# read_single_data('COS_result5.txt')
# read_single_data('COS_result10.txt')
# read_single_data('COS_result20.txt')
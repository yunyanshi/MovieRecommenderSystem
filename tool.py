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

read_single_data('result5_Pearson.txt')
read_single_data('result10_Pearson.txt')
read_single_data('result20_Pearson.txt')
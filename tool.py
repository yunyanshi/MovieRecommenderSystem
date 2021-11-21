def read_single_data(data_path):
    # Use a breakpoint in the code line below to debug your script.
    file = open(data_path, 'r')
    lines = file.readlines()
    data = []
    count = 0
    for line in lines:
        ratings = line.split()
        ratings = [int(rating) for rating in ratings]
        rating = ratings[2]
        if rating <= 0 or rating > 5:
            # print(ratings)
            count += 1
    print(count)

# read_single_data('Pearson_result5.txt')
# read_single_data('Pearson_result10.txt')
# read_single_data('Pearson_result20.txt')


# read_single_data('COS_result5.txt')
# read_single_data('COS_result10.txt')
# read_single_data('COS_result20.txt')

read_single_data('ItemBased_result5.txt')
read_single_data('ItemBased_result10.txt')
read_single_data('ItemBased_result20.txt')


def reorder_results(reference_path, result_path, new_result_path):
    result_file = open(result_path, 'r')
    lines = result_file.readlines()
    result = {}
    for line in lines:
        ratings = line.split()
        uid, mid, rating = ratings
        result[uid + ' ' + mid] = rating

    reference_file = open(reference_path, 'r')
    lines = reference_file.readlines()
    new_result = []
    for line in lines:
        ratings = line.split()
        uid, mid, rating = ratings
        key = uid + ' ' + mid
        rating = result[key]
        new_result.append(key + ' ' + rating)

    with open(new_result_path, "a") as myfile:
        myfile.write("\n".join(new_result))

# reorder_results('Pearson_result5.txt', 'test5_result.txt', 'test5_result_reordered.txt')
# reorder_results('Pearson_result10.txt', 'test10_result.txt', 'test10_result_reordered.txt')
# reorder_results('Pearson_result20.txt', 'test20_result.txt', 'test20_result_reordered.txt')


def compare_two_results(path1, path2):
    file1 = open(path1, 'r')
    lines1 = file1.readlines()

    file2 = open(path2, 'r')
    lines2 = file2.readlines()

    if len(lines1) != len(lines2):
        print("len(lines1) != len(lines2)")
        return
    count = 0
    same = 0
    for i in range(len(lines1)):
        line1, line2 = lines1[i], lines2[i]
        if line1 != line2:
            count += 1
            # print(line1 + line2)
        else:
            same += 1
            # print("line1 != line2")
            # return
    print(count)
    print(same)
    print("Have same content")

# compare_two_results('ItemBased_result5_v0.txt', 'ItemBased_result5.txt')
# compare_two_results('ItemBased_result10_v0.txt', 'ItemBased_result10.txt')
# compare_two_results('ItemBased_result20_v0.txt', 'ItemBased_result20.txt')




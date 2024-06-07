def problem_p02606(input_data):
    l, r, d = [int(i) for i in input_data.split()]

    count = 0

    for i in range(l, r + 1):

        if i % d == 0:

            count += 1

    return count

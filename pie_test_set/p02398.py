def problem_p02398(input_data):
    a, b, c = list(map(int, input_data.split()))

    count = 0

    for i in range(a, b + 1):

        if c % i == 0:

            count += 1

    return count

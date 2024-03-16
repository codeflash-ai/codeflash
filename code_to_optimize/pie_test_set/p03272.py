def problem_p03272(input_data):
    N, i = list(map(int, input_data.split()))

    count = 0

    while N >= i:

        count += 1

        N -= 1

    return count

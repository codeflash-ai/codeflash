def problem_p02766(input_data):
    n, k = input_data.split()

    n = int(n)

    k = int(k)

    count = 0

    while True:

        if n == 0:

            break

        else:

            n = n // k

            count += 1

    return count

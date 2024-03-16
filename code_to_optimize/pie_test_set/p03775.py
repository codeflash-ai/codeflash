def problem_p03775(input_data):
    n = int(eval(input_data))

    min_f = 10

    for i in range(1, n + 1):

        if i**2 > n:

            break

        if n % i != 0:

            continue

        min_seq = max(len(str(i)), len(str(n // i)))

        if min_seq < min_f:

            min_f = min_seq

    return min_f

def problem_p03284(input_data):
    n, k = list(map(int, input_data.split()))

    if n % k > 0:

        return 1

    else:

        return 0

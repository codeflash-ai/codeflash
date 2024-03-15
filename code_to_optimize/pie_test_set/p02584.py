def problem_p02584(input_data):
    X, K, D = list(map(int, input_data.split()))

    if X + K * D <= 0:

        return -X - K * D

    elif X - K * D >= 0:

        return X - K * D

    else:

        div = X // D

        mod = X % D

        if (K - div) % 2 == 0:

            return mod

        else:

            return abs(mod - D)

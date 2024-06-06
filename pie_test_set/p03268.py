def problem_p03268(input_data):
    N, K = list(map(int, input_data.split()))

    if K % 2 == 1:

        n = N // K

        ans = n**3

    else:

        n1 = N // K

        n2 = 1 + (N - K // 2) // K

        ans = n1**3 + n2**3

    return ans

def problem_p03420(input_data):
    n, k = [int(i) for i in input_data.split()]

    ans = 0

    if k == 0:

        ans = n * n

    else:

        for b in range(k + 1, n + 1):

            # number of perfect cycle

            ans += max(n // b, 0) * (b - k)

            r = n % b

            ans += max(r - k + 1, 0)

    return ans

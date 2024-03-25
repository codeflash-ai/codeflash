def problem_p02708(input_data):
    M = 10**9 + 7

    n, k = list(map(int, input_data.split()))

    a = 0

    for i in range(k, n + 2):

        l = i * ~-i // 2

        r = n * -~n // 2 - (n - i) * (n - i + 1) // 2

        a = (a + r + 1 - l) % M

    return a

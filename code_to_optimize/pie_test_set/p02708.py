def problem_p02708():
    M = 10**9 + 7

    n, k = list(map(int, input().split()))

    a = 0

    for i in range(k, n + 2):

        l = i * ~-i // 2

        r = n * -~n // 2 - (n - i) * (n - i + 1) // 2

        a = (a + r + 1 - l) % M

    print(a)


problem_p02708()

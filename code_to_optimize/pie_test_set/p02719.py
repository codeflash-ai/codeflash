def problem_p02719():
    N, K = list(map(int, input().split()))

    r = N % K

    print((min(r, K - r)))


problem_p02719()

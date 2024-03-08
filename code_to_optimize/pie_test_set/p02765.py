def problem_p02765():
    N, R = list(map(int, input().split()))

    print((R + max(0, 10 - N) * 100))


problem_p02765()

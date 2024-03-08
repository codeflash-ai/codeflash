def problem_p03186():
    A, B, C = list(map(int, input().split()))

    print((min(C, A + B + 1) + B))


problem_p03186()

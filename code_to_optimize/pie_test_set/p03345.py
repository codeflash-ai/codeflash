def problem_p03345():
    A, B, C, K = list(map(int, input().split()))

    print((B - A if K % 2 else A - B))


problem_p03345()

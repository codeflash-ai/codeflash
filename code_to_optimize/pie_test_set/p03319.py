def problem_p03319():
    import collections

    N, K = list(map(int, input().split()))

    A = list(map(int, input().split()))

    B = collections.Counter(A)[min(A)]

    print((-(-(N - B) // (K - 1))))


problem_p03319()

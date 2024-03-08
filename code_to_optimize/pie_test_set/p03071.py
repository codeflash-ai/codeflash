def problem_p03071():
    import heapq

    a, b = list(map(int, input().split()))

    print((sum(heapq.nlargest(2, [a, a - 1, b, b - 1]))))


problem_p03071()

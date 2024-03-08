def problem_p03035():
    a, b = list(map(int, input().split()))

    print((b if 13 <= a else b // 2 if 6 <= a else 0))


problem_p03035()

def problem_p04031():
    import statistics

    n = int(eval(input()))

    a = list(map(int, input().split()))

    num = round(statistics.mean(a))

    print((sum([(num - i) ** 2 for i in a])))


problem_p04031()

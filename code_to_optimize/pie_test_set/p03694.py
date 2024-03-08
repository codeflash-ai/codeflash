def problem_p03694():
    n = int(eval(input()))

    lst = sorted(map(int, input().split()))

    print((lst[-1] - lst[0]))


problem_p03694()

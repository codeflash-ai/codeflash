def problem_p03308():
    eval(input())

    lst = list(map(int, input().split()))

    lst.sort()

    print((lst[-1] - lst[0]))


problem_p03308()

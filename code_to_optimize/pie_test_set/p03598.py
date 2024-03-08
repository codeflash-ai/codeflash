def problem_p03598():
    _, k = eval(input()), int(eval(input()))
    print((sum([min(i, (k - i)) * 2 for i in list(map(int, input().split()))])))


problem_p03598()

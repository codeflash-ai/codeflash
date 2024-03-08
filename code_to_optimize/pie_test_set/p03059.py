def problem_p03059():
    a, b, t = list(map(int, input().split()))

    ret = 0

    for i in range(a, t + 1, a):

        ret += b

    print(ret)


problem_p03059()

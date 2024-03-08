def problem_p03380():
    n = int(eval(input()))

    A = sorted([int(i) for i in input().split()])

    A_max = A.pop()

    print((A_max, min(A, key=lambda x: abs(A_max - x * 2))))


problem_p03380()

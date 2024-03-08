def problem_p03378():
    N, M, X = list(map(int, input().split(" ")))

    A = list(map(int, input().split(" ")))

    left, right = 0, 0

    for ai in A:

        if ai < X:
            left += 1

        else:
            right += 1

    print((min(right, left)))


problem_p03378()

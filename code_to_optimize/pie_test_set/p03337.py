def problem_p03337():
    A, B = list(map(int, input().split()))

    C = [A + B, A - B, A * B]

    print((max(C)))


problem_p03337()

def problem_p03105():
    A, B, C = list(map(int, input().split()))

    print((C if B // A > C else B // A))


problem_p03105()

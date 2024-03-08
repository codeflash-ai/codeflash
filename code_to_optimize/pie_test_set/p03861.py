def problem_p03861():
    import math

    a, b, x = list(map(int, input().split()))

    print((b // x - (a - 1) // x))


problem_p03861()

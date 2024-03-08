def problem_p01741():
    import math

    n = float(eval(input()))

    print((int(n) + 1 if n * math.sqrt(2) < int(n) + 1 else n * math.sqrt(2)))


problem_p01741()

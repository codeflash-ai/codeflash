def problem_p02767():
    import numpy as np

    n = int(eval(input()))

    x = list(map(int, input().split()))

    x2 = np.power(x, 2)

    p1 = sum(x) // n

    p2 = -(-sum(x) // n)

    w1 = n * (p1**2) - 2 * sum(x) * p1 + sum(x2)

    w2 = n * (p2**2) - 2 * sum(x) * p2 + sum(x2)

    print((min(w1, w2)))


problem_p02767()

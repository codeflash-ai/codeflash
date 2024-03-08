def problem_p02940():
    import math

    n = int(eval(input()))

    a = 1
    c = {"R": 0, "G": 0, "B": 0}

    m = 998244353

    for i in eval(input()):

        s = sorted(c.values())

        c[i] += 1

        if c[i] > s[2]:
            pass

        elif c[i] > s[1]:
            a *= s[2] - s[1]

        else:
            a *= s[1] - s[0]

        a %= m

    print((math.factorial(n) * a % m))


problem_p02940()

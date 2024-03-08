def problem_p00424():
    while 1:

        n = int(eval(input()))

        if n == 0:
            break

        d = {}

        for _ in [0] * n:

            k, v = input().strip().split()

            d[k] = v

        a = ""

        for _ in [0] * int(eval(input())):

            e = input().strip()

            a += d[e] if e in d else e

        print(a)


problem_p00424()

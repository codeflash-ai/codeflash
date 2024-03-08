def problem_p00160():
    while 1:

        n = eval(input())

        if n == 0:
            break

        A = []

        for _ in [0] * n:

            x, y, h, w = list(map(int, input().split()))

            a = max(5 - (160 - x - y - h) / 20, 0)

            b = 5 - (25 - w) / 5 if w > 2 else 0

            A.append(max(a, b))

        print(sum([600 + e * 200 for e in A if e < 6]))


problem_p00160()

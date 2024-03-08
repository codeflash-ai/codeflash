def problem_p00107():
    while 1:

        a, b, c = sorted(map(int, input().split()))

        if a == b == c == 0:
            break

        D = (a * a + b * b) ** 0.5

        for _ in [0] * eval(input()):
            print(["NA", "OK"][2 * eval(input()) > D])


problem_p00107()

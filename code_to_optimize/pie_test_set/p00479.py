def problem_p00479():
    N = int(eval(input()))

    for _ in [0] * int(eval(input())):

        a, b = list(map(int, input().split()))

        print((min(a - 1, N - a, b - 1, N - b) % 3 + 1))


problem_p00479()

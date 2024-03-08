def problem_p00292():
    n = int(eval(input()))

    for _ in range(n):

        K, P = list(map(int, input().split()))

        a = K - (K // P) * P

        print((P * (a == 0) + (a) * (a != 0)))


problem_p00292()

def problem_p03564():
    N, K = int(eval(input())), int(eval(input()))

    now = 1

    for i in range(N):

        if now * 2 < now + K:

            now *= 2

        else:

            now += K

    print(now)


problem_p03564()

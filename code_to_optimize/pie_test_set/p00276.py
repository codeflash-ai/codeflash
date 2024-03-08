def problem_p00276():
    for i in range(int(eval(input()))):

        c, a, n = list(map(int, input().split()))

        cnt = 0

        while True:

            if c < 1 or a < 1 or n < 1:

                break

            c -= 1

            a -= 1

            n -= 1

            cnt += 1

        while True:

            if c < 2 or a < 1:

                break

            c -= 2

            a -= 1

            cnt += 1

        while True:

            if c < 3:

                break

            c -= 3

            cnt += 1

        print(cnt)


problem_p00276()

def problem_p01314():
    while True:

        n = eval(input())

        if n == 0:

            break

        ans = 0

        for i in range(1, n / 2 + 1):

            su = 0

            c = i

            while su < n:

                su += c

                c += 1

            if su == n:

                ans += 1

        print(ans)


problem_p01314()

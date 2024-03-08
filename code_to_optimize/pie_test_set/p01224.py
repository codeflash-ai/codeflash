def problem_p01224():
    def div(n):

        ls = []

        i = 2

        while i * i <= n:

            c = 0

            if n % i == 0:

                while n % i == 0:

                    n /= i

                    c += 1

            if c > 0:

                ls.append([i, c])

            i += 1

        if n > 1:

            ls.append([n, 1])

        ans = 1

        for b, p in ls:

            ans *= (b ** (p + 1) - 1) / (b - 1)

        return ans

    while 1:

        n = eval(input())

        if n == 0:
            break

        d = div(n) - n

        if d == n:

            print("perfect number")

        elif d < n:

            print("deficient number")

        else:

            print("abundant number")


problem_p01224()

def problem_p01094():
    # AOJ 1609: Look for the Winner!

    # Python3 2018.7.13 bal4u

    while True:

        n = int(eval(input()))

        if n == 0:
            break

        c = list(input().split())

        c = [ord(i) - ord("A") for i in c]

        f = [0] * 26

        max1 = max2 = 0

        x1 = x2 = -1

        k, i = n, 0

        while i < n:

            x = c[i]

            i, k = i + 1, k - 1

            f[x] += 1

            if f[x] > max1:

                if x1 < 0 or x == x1:
                    max1 = f[x]
                    x1 = x

                else:
                    max2 = max1
                    x2 = x1
                    max1 = f[x]
                    x1 = x

            elif f[x] > max2:
                max2 = f[x]
                x2 = x

            if max2 + k < max1:
                break

        if max1 == max2:
            print("TIE")

        else:
            print((chr(x1 + ord("A")), i))


problem_p01094()

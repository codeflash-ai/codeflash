def problem_p02320():
    def main():

        n, w = list(map(int, input().split()))

        value = []

        weight = []

        num = []

        for _ in range(n):

            vi, wi, ni = list(map(int, input().split()))

            value.append(vi)

            weight.append(wi)

            num.append(ni)

        def to_digit(x):

            acc = 1

            ret = [0]

            while x >= acc:

                ret.append(acc)

                x -= acc

                acc *= 2

            if x:

                ret.append(x)

            return ret

        num = list(map(to_digit, num))

        dp = [0 for _ in range(w + 1)]

        for i in range(n):

            vi = value[i]

            wi = weight[i]

            numsi = num[i]

            for k in numsi:

                wik = wi * k

                vik = vi * k

                for j in range(w, wik - 1, -1):

                    dp[j] = max(dp[j], dp[j - wik] + vik)

        print((dp[w]))

    main()


problem_p02320()

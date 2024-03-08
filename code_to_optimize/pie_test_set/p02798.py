def problem_p02798():

    def main():

        import sys

        input = sys.stdin.readline

        n = int(eval(input()))

        a = tuple(map(int, input().split()))

        b = tuple(map(int, input().split()))

        dg = 20

        pp = 20

        dp = [10**5] * (10**7)

        new = set([])

        for i in range(n):

            s = 1 << i

            dp[s * pp + i] = i

            new.add(s * dg + i)

        for k in range(n - 1):

            tank = set([])

            if len(new) == 0:

                break

            for popelt in new:

                s, idx = popelt // dg, popelt % dg

                if (k - idx) % 2 == 0:

                    fr = a[idx]

                else:

                    fr = b[idx]

                cnt = 0

                for j in range(n):

                    if (s >> j) & 1 == 0:

                        if (j - k) % 2 == 1:

                            val = a[j]

                        else:

                            val = b[j]

                        if val >= fr and dp[(s + (1 << j)) * pp + j] > dp[s * pp + idx] + j - cnt:

                            dp[(s + (1 << j)) * pp + j] = dp[s * pp + idx] + j - cnt

                            tank.add((s + (1 << j)) * dg + j)

                    else:

                        cnt += 1

            new = tank

        res = 10**5

        s = (2**n - 1) * pp

        for i in range(n):

            if res > dp[s + i]:

                res = dp[s + i]

        if res == 10**5:

            print((-1))

        else:

            print(res)

    if __name__ == "__main__":

        main()


problem_p02798()

def problem_p02651():
    # 一次独立：rankが成分数

    import numpy as np

    # 0-indexed,MSBがma以下の時

    def msb(x, ma=59):

        for i in range(ma, -1, -1):

            if (x >> i) & 1:

                return i

        return len()

    t = int(eval(input()))

    for _ in range(t):

        n = int(eval(input()))

        a = list(map(int, input().split()))

        s = eval(input())

        # 独立、msb同じものはない

        # 逆に異なれば独立

        msbs = [0 for i in range(60)]

        for i in range(n - 1, -1, -1):

            now = a[i]

            m = 59

            # 独立のときfはTrue、独立でない時fはFalse

            while True:

                m = msb(now, m)

                if msbs[m] == 0:

                    f = True

                    break

                now = msbs[m] ^ now

                if now == 0:

                    f = False

                    break

            if s[i] == "0":

                if f:

                    msbs[m] = now

            else:

                if f:

                    print((1))

                    break

        else:

            print((0))


problem_p02651()

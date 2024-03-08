def problem_p03318():
    #!/usr/bin/env python

    # coding: utf-8

    def ri():

        return int(eval(input()))

    def rl():

        return list(input().split())

    def rli():

        return list(map(int, input().split()))

    def calc(n):

        s = sum(map(int, str(n)))

        return n / s

    def f(n):

        ld = list(map(int, str(n)))

        cand = []

        for i in range(len(ld) - 1):

            d = len(ld) - 1 - i

            ld[d] = 9

            for i in range(1, 10):

                if i < ld[d - 1]:

                    continue

                ld[d - 1] = i

                p = int("".join(map(str, ld)))

                cand.append((calc(p), p))

        cand.sort()

        return cand[0][1]

    MAX = 10**15

    def main():

        k = ri()

        cand = list(range(1, 10))

        n = 10

        while True:

            if n > MAX:

                break

            p = f(n)

            cand.append(p)

            n = p + 1

        print(("\n".join(map(str, cand[:k]))))

    if __name__ == "__main__":

        main()


problem_p03318()

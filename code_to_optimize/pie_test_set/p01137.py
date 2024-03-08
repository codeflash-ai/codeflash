def problem_p01137():
    import math

    def solve(e):

        k = 2**32

        for z in range(100, -1, -1):

            z3 = z * z * z

            if z3 > e:

                continue

            e2 = e - z3

            ylm = int(math.sqrt(e2))

            xzlm = 3 * z * z + 3 * z + 1

            for y in range(ylm, -1, -1):

                y2 = y * y

                if e2 > (y + 1) * (y + 1):

                    break

                e3 = e2 - y2

                xylm = 2 * y + 1

                x = e3

                if x > xylm or x > xzlm:

                    continue

                k = min(k, x + y + z)

        return k

    def main():

        while True:

            a = int(eval(input()))

            if a == 0:

                break

            print((solve(a)))

    if __name__ == "__main__":

        main()


problem_p01137()

def problem_p03741():
    # ARC072C - Sequence (ABC059C)

    def main():

        n = int(eval(input()))

        lst = list(map(int, input().rstrip().split()))

        cnt = [0] * 2

        # start with positive number

        cur = 0

        for i, j in enumerate(lst):

            cur += j

            if i % 2 == 0 and cur <= 0:

                cnt[0] += abs(cur) + 1

                cur = 1

            elif i % 2 == 1 and cur >= 0:

                cnt[0] += abs(cur) + 1

                cur = -1

        # start with negative number

        cur = 0

        for i, j in enumerate(lst):

            cur += j

            if i % 2 == 0 and cur >= 0:

                cnt[1] += abs(cur) + 1

                cur = -1

            elif i % 2 == 1 and cur <= 0:

                cnt[1] += abs(cur) + 1

                cur = 1

        print((min(cnt)))

    if __name__ == "__main__":

        main()


problem_p03741()

def problem_p03156():
    def main():

        N = int(eval(input()))

        A, B = (int(i) for i in input().split())

        P = [int(i) for i in input().split()]

        cnt = [0, 0, 0]

        for p in P:

            if p <= A:

                cnt[0] += 1

            elif p <= B:

                cnt[1] += 1

            else:

                cnt[2] += 1

        print((min(cnt)))

    if __name__ == "__main__":

        main()


problem_p03156()

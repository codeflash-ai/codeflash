def problem_p00206():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0206



    """

    import sys

    from sys import stdin

    input = stdin.readline

    def main(args):

        while True:

            L = int(eval(input()))

            if L == 0:

                break

            ans = "NA"

            for i in range(1, 12 + 1):

                M, N = list(map(int, input().split()))

                if ans == "NA":

                    L -= M - N

                    if L <= 0:

                        ans = i

            print(ans)

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00206()

def problem_p00181():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0181



    """

    import sys

    from sys import stdin

    input = stdin.readline

    def Cond(m, n, mid, books):

        if max(books) > mid:

            return False

        rem = mid

        while books:

            while rem >= books[0]:

                rem -= books[0]

                books = books[1:]

                if len(books) == 0:

                    break

            rem = mid

            m -= 1

        if m >= 0:

            return True

        else:

            return False

    def solve(m, n, books):

        ub = 1500000

        lb = 0

        min_width = float("inf")

        for i in range(50):

            mid = (ub + lb) // 2

            if Cond(m, n, mid, books):

                min_width = min(min_width, mid)

                ub = mid

            else:

                lb = mid

        return min_width

    def main(args):

        while True:

            m, n = list(map(int, input().split()))

            if m == 0 and n == 0:

                break

            books = [int(eval(input())) for _ in range(n)]

            result = solve(m, n, books)

            print(result)

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00181()

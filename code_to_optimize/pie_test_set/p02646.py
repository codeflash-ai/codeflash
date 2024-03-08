def problem_p02646():
    def main():

        A, V = list(map(int, input().split()))

        B, W = list(map(int, input().split()))

        T = int(eval(input()))

        if A == B:

            print("YES")

            return

        d = abs(A - B)

        dv = V - W

        if d <= dv * T:

            print("YES")

        else:

            print("NO")

        return

    if __name__ == "__main__":

        main()

    # import sys

    # input = sys.stdin.readline

    #

    # sys.setrecursionlimit(10 ** 7)

    #

    # (int(x)-1 for x in input().split())

    # rstrip()

    #

    # def binary_search(*, ok, ng, func):

    #     while abs(ok - ng) > 1:

    #         mid = (ok + ng) // 2

    #         if func(mid):

    #             ok = mid

    #         else:

    #             ng = mid

    #     return ok


problem_p02646()

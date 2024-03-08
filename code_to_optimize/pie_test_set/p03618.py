def problem_p03618():
    import sys

    def input():
        return sys.stdin.readline().rstrip()

    from collections import defaultdict

    def main():

        A = eval(input())

        len_a = len(A)

        dp = [0] * len_a

        dp[0] = 1

        dd = defaultdict(int)

        dd[A[0]] += 1

        for i in range(1, len_a):

            a = A[i]

            dp[i] = i - dd[a]

            dd[a] += 1

        print((sum(dp)))

    if __name__ == "__main__":

        main()


problem_p03618()

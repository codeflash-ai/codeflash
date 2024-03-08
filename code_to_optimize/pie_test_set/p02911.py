def problem_p02911():
    import sys

    input = sys.stdin.readline

    def main():

        N, K, Q = list(map(int, input().split()))

        a = [K - Q] * N

        for _ in range(Q):

            a[int(eval(input())) - 1] += 1

        for i in range(N):

            if a[i] > 0:

                print("Yes")

            else:

                print("No")

    if __name__ == "__main__":

        main()


problem_p02911()

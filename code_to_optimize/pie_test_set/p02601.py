def problem_p02601():
    def main():

        import sys

        A, B, C = list(map(int, sys.stdin.readline().split()))

        K = int(sys.stdin.readline())

        for i in range(K + 1):

            if B <= A:

                B *= 2

            elif C <= B:

                C *= 2

            else:

                print("Yes")

                return

        print("No")

    main()


problem_p02601()

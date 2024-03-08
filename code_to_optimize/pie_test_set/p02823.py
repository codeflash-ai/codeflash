def problem_p02823():
    import sys

    read = sys.stdin.buffer.read

    input = sys.stdin.buffer.readline

    inputs = sys.stdin.buffer.readlines

    # mod=10**9+7

    # rstrip().decode('utf-8')

    # map(int,input().split())

    # import numpy as np

    def main():

        n, a, b = list(map(int, input().split()))

        if (b - a) % 2 == 0:

            print(((b - a) // 2))

        else:

            ma = max(a, b)

            mi = min(a, b)

            ko1 = min(n - mi, ma - 1)

            ko2 = n - ma + 1 + (n - (mi + (n - ma + 1))) // 2

            ko3 = mi - 1 + 1 + ((ma - (mi - 1 + 1)) - 1) // 2

            print((min(ko1, ko2, ko3)))

    if __name__ == "__main__":

        main()


problem_p02823()

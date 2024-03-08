def problem_p02839():
    import sys

    read = sys.stdin.buffer.read

    input = sys.stdin.buffer.readline

    inputs = sys.stdin.buffer.readlines

    # mod=10**9+7

    # rstrip().decode('utf-8')

    # map(int,input().split())

    # import numpy as np

    def main():

        h, w = list(map(int, input().split()))

        G = [[0] * w for i in range(h)]

        A = [0] * h

        B = [0] * h

        for i in range(h):

            A[i] = list(map(int, input().split()))

        for i in range(h):

            B[i] = list(map(int, input().split()))

        for i in range(h):

            for j in range(w):

                G[i][j] = abs(A[i][j] - B[i][j])

        M = (h + w) * 80

        li = [[[0] * M for i in range(w)] for j in range(h)]

        li[0][0][G[0][0]] = 1

        for i in range(h):

            for j in range(w):

                for k in range(M):

                    if li[i][j][k] == 1:

                        if i < h - 1:

                            li[i + 1][j][abs(k - G[i + 1][j])] = 1

                            li[i + 1][j][abs(k + G[i + 1][j])] = 1

                        if j < w - 1:

                            li[i][j + 1][abs(k - G[i][j + 1])] = 1

                            li[i][j + 1][abs(k + G[i][j + 1])] = 1

        for i in range(M):

            if li[h - 1][w - 1][i] == 1:

                print(i)

                exit(0)

    if __name__ == "__main__":

        main()


problem_p02839()

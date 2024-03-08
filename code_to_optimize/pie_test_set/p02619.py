def problem_p02619():
    import bisect

    import random

    import sys

    input = sys.stdin.readline

    class score:

        def __init__(self, t):

            self.t = t

        def first_score(self):

            last = [0] * 26

            score = 0

            for i, ti in enumerate(self.t):

                score += S[i][ti - 1]

                last[ti - 1] = i + 1

                for j in range(26):

                    score -= C[j] * ((i + 1) - last[j])

            self.score = score

            return score

    D = int(eval(input()))

    C = list(map(int, input().split()))

    S = [list(map(int, input().split())) for i in range(D)]

    # T = [int(input()) for i in range(D)]

    T = []

    for i in range(D):

        T.append(int(eval(input())))

        s = score(T)

        print((s.first_score()))


problem_p02619()

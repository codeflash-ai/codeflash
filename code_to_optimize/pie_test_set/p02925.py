def problem_p02925():
    import sys

    import collections

    import copy

    input = sys.stdin.readline

    def main():

        N = int(eval(input()))

        A = [collections.deque([int(x) for x in input().split()]) for _ in range(N)]

        day = 0

        c = 0

        nxt = [x for x in range(N)]

        oppo = [0] * N

        while c != N * (N - 1):

            day += 1

            for i in nxt:

                oppo[i] = A[i].popleft() if A[i] else -1

            tmp = set()

            for i in nxt:

                if i == -1 or i in tmp:

                    continue

                if i == oppo[oppo[i] - 1] - 1:

                    tmp.add(i)

                    tmp.add(oppo[i] - 1)

                    c += 2

            if not tmp:

                print((-1))

                return

            nxt = tmp

        print(day)

    if __name__ == "__main__":

        main()


problem_p02925()

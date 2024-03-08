def problem_p01288():
    def solve():

        import sys

        file_input = sys.stdin

        while True:

            N, Q = list(map(int, file_input.readline().split()))

            if N == 0:

                break

            parent = [None, None]

            parent += [int(file_input.readline()) for _ in range(N - 1)]

            unmarked = [True] * (N + 1)

            unmarked[1] = False

            ans = 0

            for _ in range(Q):

                line = file_input.readline()

                v = int(line[2:])

                if line[0] == "M":

                    unmarked[v] = False

                else:

                    while unmarked[v]:

                        v = parent[v]

                    ans += v

            print(ans)

    solve()


problem_p01288()

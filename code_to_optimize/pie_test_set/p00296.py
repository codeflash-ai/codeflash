def problem_p00296():
    def solve():

        import sys

        file_input = sys.stdin

        N, M, Q = map(int, file_input.readline().split())

        student = list(range(N))

        exemption = [1] * N

        pos = 0

        for a in map(int, file_input.readline().split()):

            if a % 2:

                pos = (pos - a) % len(student)

            else:

                pos = (pos + a) % len(student)

            s = student[pos]

            exemption[s] = 0

            del student[pos]

        question = map(int, file_input.readline().split())

        print(*map(lambda x: exemption[x], question), sep="\n")

    solve()


problem_p00296()

def problem_p00008():
    import sys

    def dfs(a, b):

        if a == 0:
            return 1

        elif b == 0 or a < 0:
            return 0

        return sum([dfs(a - i, b - 1) for i in range(10)])

    for line in sys.stdin.readlines():

        print(dfs(int(line.strip()), 4))


problem_p00008()

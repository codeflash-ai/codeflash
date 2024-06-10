def problem_p02786(input_data):
    from functools import lru_cache

    import sys

    sys.setrecursionlimit(10**7)

    @lru_cache(maxsize=None)
    def dfs(h):

        if h == 1:

            return 1

        return 1 + dfs(h // 2) * 2

    def main():

        h = int(eval(input_data))

        return dfs(h)

    if __name__ == "__main__":

        main()

    #

    # input = sys.stdin.readline

    # rstrip()

    # int(input_data)

    # map(int, input_data.split())

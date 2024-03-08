def problem_p02657():
    def iput():
        return int(eval(input()))

    def mput():
        return list(map(int, input().split()))

    def lput():
        return list(map(int, input().split()))

    def solve():

        a, b = mput()

        print((a * b))

        return 0

    if __name__ == "__main__":

        solve()


problem_p02657()

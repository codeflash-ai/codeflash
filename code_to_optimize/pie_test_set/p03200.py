def problem_p03200():
    import sys

    sys.setrecursionlimit(10**6)

    if sys.platform in (["ios", "darwin", "win32"]):

        sys.stdin = open("Untitled.txt")

    input = sys.stdin.readline

    def INT():
        return int(eval(input()))

    def MAP():
        return [int(s) for s in input().split()]

    def main():

        S = input().rstrip()

        bcnt = 0

        A = []

        for i in range(len(S)):

            if S[i] == "B":
                bcnt += 1

            if S[i] == "W":
                A.append(bcnt)

        print((sum(A)))

    if __name__ == "__main__":

        main()


problem_p03200()

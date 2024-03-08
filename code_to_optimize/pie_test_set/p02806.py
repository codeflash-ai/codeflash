def problem_p02806():
    import sys

    from pprint import pprint

    def solve(n, s, t, x):

        ans = 0

        slept = False

        for i in range(n):

            if slept:

                ans += t[i]

            if s[i] == x:

                slept = True

        print(ans)

    if __name__ == "__main__":

        n = int(sys.stdin.readline().strip())

        s = [""] * n

        t = [0] * n

        for i in range(n):

            s[i], tmp = sys.stdin.readline().strip().split(" ")

            t[i] = int(tmp)

        x = sys.stdin.readline().strip()

        solve(n, s, t, x)


problem_p02806()

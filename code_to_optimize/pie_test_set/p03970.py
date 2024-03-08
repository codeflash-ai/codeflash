def problem_p03970():

    import sys

    def input():
        return sys.stdin.readline().strip()

    def resolve():

        s = eval(input())

        t = "CODEFESTIVAL2016"

        cnt = 0

        for i in range(16):

            if s[i] != t[i]:

                cnt += 1

        print(cnt)

    resolve()


problem_p03970()

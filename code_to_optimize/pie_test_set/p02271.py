def problem_p02271():
    n = int(eval(input()))

    A = list(map(int, input().split()))

    r = []

    def f(s, k):

        if k >= 0:

            global r

            r += [s]

            f(s + A[k - 1], k - 1)

            f(s, k - 1)

    f(0, n)

    eval(input())

    for e in map(int, input().split()):
        print((["no", "yes"][e in r]))


problem_p02271()

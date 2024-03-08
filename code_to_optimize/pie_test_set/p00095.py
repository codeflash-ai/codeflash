def problem_p00095():
    n = int(input())

    mxs, minm = 0, 1000

    for i in range(n):

        m, s = list(map(int, input().split()))

        if s > mxs:

            mxs = s

            minm = m

        elif s == mxs:

            if m < minm:

                minm = m

    print(minm, mxs)


problem_p00095()

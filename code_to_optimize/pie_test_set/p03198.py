def problem_p03198():
    #

    import collections, atexit, math, sys, bisect

    sys.setrecursionlimit(1000000)

    def getIntList():

        return list(map(int, input().split()))

    try:

        # raise ModuleNotFoundError

        import numpy

        def dprint(*args, **kwargs):

            # print(*args, **kwargs, file=sys.stderr)

            # in python 3.4 **kwargs is invalid???

            print(*args, file=sys.stderr)

        dprint("debug mode")

    except Exception:

        def dprint(*args, **kwargs):

            pass

    inId = 0

    outId = 0

    if inId > 0:

        dprint("use input", inId)

        sys.stdin = open("input" + str(inId) + ".txt", "r")  # 标准输出重定向至文件

    if outId > 0:

        dprint("use output", outId)

        sys.stdout = open("stdout" + str(outId) + ".txt", "w")  # 标准输出重定向至文件

        atexit.register(lambda: sys.stdout.close())  # idle 中不会执行 atexit

    if True:

        (N,) = getIntList()

        # print(N)

        za = getIntList()

    else:

        N = 1000

        import random

        za = [random.randint(1, 1000000000) for i in range(N)]

    dprint("begin")

    zleft = [0 for i in range(N + 1)]

    zright = [0 for i in range(N + 1)]

    def getwork(zr, za):

        zr[0] = 0

        st = []

        for i in range(N):

            if i % 10000 == 0:

                dprint("---", i)

            nt = [[za[i], 0], [za[i], 0], 1]

            st.append(nt)

            nr = zr[i]

            # dprint(st)

            while len(st) > 1:

                b_2 = st[-2][1][1]

                b_1 = st[-1][0][1]

                bb = min(b_1, b_2)

                b_1 -= bb

                b_2 -= bb

                b_1p = 4**b_1

                b_2p = 4**b_2

                t2 = st[-2][1][0] * b_2p

                t1 = st[-1][0][0] * b_1p

                if t2 < t1:

                    nr += 2 * st[-2][2]

                    st[-2][0][1] += 1

                    st[-2][1][1] += 1

                elif t2 < t1 * 4:

                    ttt = [st[-2][0], st[-1][1], st[-2][2] + st[-1][2]]

                    st[-2] = ttt

                    st.pop()

                else:

                    break

            # dprint(st)

            zr[i + 1] = nr

            # dprint(nr)

            # dprint(st)

            assert nr < 1000000000000000

    getwork(zleft, za)

    dprint("!!!!")

    getwork(zright, za[::-1])

    # dprint(zleft)

    # dprint(zright)

    r = zright[N]

    for i in range(N + 1):

        tr = i + zleft[i] + zright[N - i]

        r = min(r, tr)

    print(r)

    dprint("end")


problem_p03198()

def problem_p03223():
    import sys

    stdin = sys.stdin

    sys.setrecursionlimit(10**5)

    def li():
        return list(map(int, stdin.readline().split()))

    def li_():
        return [int(x) - 1 for x in stdin.readline().split()]

    def lf():
        return list(map(float, stdin.readline().split()))

    def ls():
        return stdin.readline().split()

    def ns():
        return stdin.readline().rstrip()

    def lc():
        return list(ns())

    def ni():
        return int(stdin.readline())

    def nf():
        return float(stdin.readline())

    from collections import deque

    n = ni()

    a = []

    for _ in range(n):

        a.append(ni())

    mx_cent = [ai for ai in a]

    mn_cent = [ai for ai in a]

    a.sort()

    a = deque(a)

    mx_cent = deque([ai for ai in a])

    mn_cent = deque([ai for ai in a])

    bmx = deque([mx_cent.pop()])

    bmn = deque([mn_cent.popleft()])

    cnt = 0

    while len(mx_cent) > 1:

        lf = 0

        rt = 0

        if cnt % 2 == 0:

            lf = mx_cent.popleft()

            rt = mx_cent.popleft()

        else:

            lf = mx_cent.pop()

            rt = mx_cent.pop()

        bmx.appendleft(lf)

        bmx.append(rt)

        if cnt % 2 == 0:

            lf = mn_cent.pop()

            rt = mn_cent.pop()

        else:

            lf = mn_cent.popleft()

            rt = mn_cent.popleft()

        bmn.appendleft(lf)

        bmn.append(rt)

        cnt += 1

    if len(mx_cent) == 1:

        last = mx_cent.pop()

        if abs(bmx[0] - last) > abs(bmx[-1] - last):

            bmx.appendleft(last)

        else:

            bmx.append(last)

        last = mn_cent.pop()

        if abs(bmn[0] - last) > abs(bmn[-1] - last):

            bmn.appendleft(last)

        else:

            bmn.append(last)

    diffmx, diffmn = 0, 0

    for i in range(n - 1):

        diffmx += abs(bmx[i + 1] - bmx[i])

        diffmn += abs(bmn[i + 1] - bmn[i])

    print((max(diffmx, diffmn)))


problem_p03223()

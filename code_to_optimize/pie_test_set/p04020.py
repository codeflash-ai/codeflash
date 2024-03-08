def problem_p04020():
    import sys

    stdin = sys.stdin

    sys.setrecursionlimit(10**8)

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

    n = ni()

    a = [ni() for _ in range(n)]

    cnt = {i + 1: ai for i, ai in enumerate(a)}

    prev = -100

    cur = 0

    ans = 0

    for key in sorted(cnt.keys()):

        if cnt[key] == 0:

            continue

        if prev == -100:

            prev = key

        if (key - prev) < 2:

            cur += cnt[key]

        else:

            cur = cnt[key]

        ans += cur // 2

        cur -= 2 * (cur // 2)

        prev = key

    print(ans)


problem_p04020()

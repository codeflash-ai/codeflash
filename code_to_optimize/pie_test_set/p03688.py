def problem_p03688():
    import sys

    stdin = sys.stdin

    def li():
        return map(int, stdin.readline().split())

    def li_():
        return map(lambda x: int(x) - 1, stdin.readline().split())

    def lf():
        return map(float, stdin.readline().split())

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

    a = list(li())

    exist = True

    if max(a) - min(a) > 1:

        exist = False

    elif max(a) >= n:

        exist = False

    elif max(a) - min(a) == 1:

        if a.count(min(a)) < max(a) and 2 * (max(a) - a.count(min(a))) <= a.count(max(a)):

            exist = True

        else:

            exist = False

    elif max(a) == min(a):

        if 2 * a[0] <= n or a[0] + 1 == n:

            exist = True

        else:

            exist = False

    print("Yes") if exist else print("No")


problem_p03688()

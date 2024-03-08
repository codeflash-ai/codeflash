def problem_p03637():
    N = int(eval(input()))

    src = list(map(int, input().split()))

    x1 = x2 = x4 = 0

    for a in src:

        if a % 2 == 1:

            x1 += 1

        elif a % 4 == 0:

            x4 += 1

        else:

            x2 += 1

    def solve():

        if x4 >= x1:
            return True

        if x4 + 1 == x1:

            return x2 == 0

        else:
            return False

    print(("Yes" if solve() else "No"))


problem_p03637()

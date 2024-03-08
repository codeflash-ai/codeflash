def problem_p03481():
    import sys

    input = sys.stdin.readline

    x, y = [int(x) for x in input().split()]

    ans = 0

    while x <= y:

        x *= 2

        ans += 1

    print(ans)


problem_p03481()

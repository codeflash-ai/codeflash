def problem_p03975():
    n, a, b = list(map(int, input().split()))

    c = 0

    for _ in range(n):

        t = int(eval(input()))

        if t < a or b <= t:

            c += 1

    print(c)


problem_p03975()

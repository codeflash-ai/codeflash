def problem_p03062():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    b = []

    minus = 0

    for i in range(n):

        if a[i] < 0:

            minus += 1

        b.append(abs(a[i]))

    b.sort()

    if minus % 2 == 0:

        print((sum(b)))

    else:

        print((sum(b) - 2 * b[0]))


problem_p03062()

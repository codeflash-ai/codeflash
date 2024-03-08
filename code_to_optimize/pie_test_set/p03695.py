def problem_p03695():
    import sys

    n = int(eval(input()))

    a = list(map(int, input().split()))

    l = []

    free = 0

    for i in range(n):

        if 1 <= a[i] <= 399:

            l.append(1)

        elif 400 <= a[i] <= 799:

            l.append(2)

        elif 800 <= a[i] <= 1199:

            l.append(3)

        elif 1200 <= a[i] <= 1599:

            l.append(4)

        elif 1600 <= a[i] <= 1999:

            l.append(5)

        elif 2000 <= a[i] <= 2399:

            l.append(6)

        elif 2400 <= a[i] <= 2799:

            l.append(7)

        elif 2800 <= a[i] <= 3199:

            l.append(8)

        else:

            free += 1

    if len(set(l)) == 0:

        print((1, free))

        sys.exit()

    print((len(set(l)), len(set(l)) + free))


problem_p03695()

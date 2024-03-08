def problem_p02412():
    while True:

        n, x = [int(i) for i in input().split()]

        if n == x == 0:

            break

        count = 0

        for s in range(1, n - 1):

            for m in range(s + 1, n):

                for e in range(m + 1, n + 1):

                    if x == sum([s, m, e]):

                        count += 1

        print(count)


problem_p02412()

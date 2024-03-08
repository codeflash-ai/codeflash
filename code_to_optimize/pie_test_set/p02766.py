def problem_p02766():
    n, k = input().split()

    n = int(n)

    k = int(k)

    count = 0

    while True:

        if n == 0:

            break

        else:

            n = n // k

            count += 1

    print(count)


problem_p02766()

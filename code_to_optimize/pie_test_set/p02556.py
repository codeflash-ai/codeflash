def problem_p02556():
    n = int(eval(input()))

    max1 = -1 * (10**10)

    min1 = 10**10

    max0 = -1 * (10**10)

    min0 = 10**10

    for i in range(n):

        x, y = list(map(int, input().split()))

        k0 = x - y

        k1 = x + y

        if k0 > max0:

            max0 = k0

        if k0 < min0:

            min0 = k0

        if k1 > max1:

            max1 = k1

        if k1 < min1:

            min1 = k1

    ans = int(max(max1 - min1, max0 - min0))

    print(ans)


problem_p02556()

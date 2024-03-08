def problem_p04014():
    n = int(eval(input()))

    s = int(eval(input()))

    if s == 1 and n == 1:

        print((2))

        exit()

    if s > n:

        print((-1))

        exit()

    if n % 2 == 0 and n > s > n // 2:

        print((-1))

        exit()

    if n % 2 == 1 and n > s > n // 2 + 1:

        print((-1))

        exit()

    ans = float("INF")

    for i in range(2, int(n**0.5) + 1):

        x = n

        count = 0

        while x >= i:

            count += x % i

            x //= i

        if count + x == s:

            print(i)

            exit()

    now = 2

    li = int(n**0.5) + 1

    while True:

        x = n // now

        if x < li:

            break

        count = 0

        y = n

        while y >= x:

            count += y % x

            y //= x

        count += y

        if count <= s and (s - count) % now == 0:

            z = x - n // (now + 1)

            if (s - count) // now < z:

                ans = x - (s - count) // now

        now += 1

    if s == n:

        print((min(ans, n + 1)))

    else:

        print((min(ans, n - s + 1)))


problem_p04014()

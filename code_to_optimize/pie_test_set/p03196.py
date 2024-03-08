def problem_p03196():
    n, p = list(map(int, input().split()))

    if n == 1:

        print(p)

    elif n > 40:

        print((1))

    else:

        ans = 1

        i = 1

        while True:

            if i**n > p:

                break

            if p % (i**n) == 0:

                ans = i

            i += 1

        print(ans)


problem_p03196()

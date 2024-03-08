def problem_p00357():
    n = int(eval(input()))

    d = [int(eval(input())) for i in range(n)]

    pos = 0

    ans = "yes"

    for i in range(n):

        if pos >= i * 10:

            pos = max(pos, i * 10 + d[i])

        else:

            ans = "no"

            break

        if pos >= (n - 1) * 10:

            break

    pos = 0

    for i in range(n):

        if pos >= i * 10:

            pos = max(pos, i * 10 + d[n - 1 - i])

        else:

            ans = "no"

            break

        if pos >= (n - 1) * 10:

            break

    print(ans)


problem_p00357()

def problem_p02833():
    N = int(eval(input()))

    if N % 2 == 1:

        print((0))

        exit()

    ans = 0

    mod = 10

    while mod <= N:

        ans += N // mod

        mod *= 5

    print(ans)


problem_p02833()

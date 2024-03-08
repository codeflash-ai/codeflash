def problem_p03951():
    N = int(eval(input()))

    s = eval(input())

    t = eval(input())

    ans = 2 * N

    for i in range(1, N + 1):

        # print(s[-i:], t[:i])

        if s[-i:] == t[:i]:

            ans = min(ans, 2 * N - i)

    print(ans)


problem_p03951()

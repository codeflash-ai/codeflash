def problem_p02983():
    mod = 2019

    l, r = list(map(int, input().split()))

    if r // mod - l // mod > 0:

        print((0))

    else:

        l %= mod

        r %= mod

        ans = mod

        for i in range(l + 1, r + 1):

            for j in range(l, i):

                ans = min(i * j % mod, ans)

            if ans == 0:

                break

        print(ans)


problem_p02983()

def problem_p02712():
    N = int(eval(input()))

    ans = 0

    for i in range(1, N + 1):

        if i % 3 == 0 or i % 5 == 0:

            ans += 0

        else:

            ans += i

    print(ans)


problem_p02712()

def problem_p03893():
    x = int(input())

    x -= 1

    ans = 6

    for i in range(x):

        ans = ans + ans + 2

    print(ans)


problem_p03893()

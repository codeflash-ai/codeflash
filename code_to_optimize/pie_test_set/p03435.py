def problem_p03435():
    c = [list(map(int, input().split())) for i in range(3)]

    res = 0

    ans = 0

    for i in range(3):

        res += sum(c[i])

        ans += c[i][i] * 3

    if res == ans:

        print("Yes")

    else:

        print("No")


problem_p03435()

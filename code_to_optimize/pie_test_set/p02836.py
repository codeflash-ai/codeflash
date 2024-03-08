def problem_p02836():
    s = eval(input())

    res = 0

    for i in range(len(s) // 2):

        res += int(s[i] != s[len(s) - i - 1])

    print(res)


problem_p02836()

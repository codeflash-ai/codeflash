def problem_p03211():
    s = eval(input())

    ans = 10 * 100

    for i in range(len(s) - 2):

        if abs(753 - int(s[i : i + 3])) <= ans:

            ans = abs(753 - int(s[i : i + 3]))

    print(ans)


problem_p03211()

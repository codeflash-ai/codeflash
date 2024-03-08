def problem_p03672():
    s = input()[:-1]

    for i in range(len(s)):

        x = s[:i]

        n = len(s[:i])

        if n % 2 == 0 and x[: i // 2] == x[i // 2 :]:

            ans = n

    print(ans)


problem_p03672()

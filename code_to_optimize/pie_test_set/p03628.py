def problem_p03628():
    n = int(eval(input()))

    s = eval(input()) + "@"

    t = eval(input()) + "@"

    mod = 10**9 + 7

    # 情報を1次元にして処理しやすくするぜ

    # 縦で一致 True 横で一致 False

    b = []

    for i in range(n):

        if s[i] == s[i + 1]:

            continue

        b.append(s[i] == t[i])

    # 処理するぜ

    if b[0]:

        ans = 3

    else:

        ans = 6

    for i in range(1, len(b)):

        if b[i]:

            if b[i - 1]:

                ans *= 2

        else:

            if b[i - 1]:

                ans *= 2

            else:

                ans *= 3

        ans %= mod

    print(ans)


problem_p03628()

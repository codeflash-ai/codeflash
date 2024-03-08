def problem_p02419():
    W = input().lower()

    s = []

    while True:

        T = list(map(str, input().split()))

        if T[0] == "END_OF_TEXT":

            break

        else:

            for i in range(len(T)):

                s.append(T[i].lower())

    ans = 0

    for i in range(len(s)):

        if s[i] == W:

            ans += 1

    print(ans)


problem_p02419()

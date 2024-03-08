def problem_p03339():
    n = int(eval(input()))

    s = eval(input())

    w = [0] + []

    e = [] + [0]

    w_cnt = 0

    e_cnt = 0

    for i in range(n):

        if s[i] == "W":

            w_cnt += 1

        if s[n - i - 1] == "E":

            e_cnt += 1

        w.append(w_cnt)

        e.append(e_cnt)

    e = e[::-1]

    ans = 10**9

    for i in range(n):

        ans = min(ans, w[i] + e[i + 1])

    print(ans)


problem_p03339()

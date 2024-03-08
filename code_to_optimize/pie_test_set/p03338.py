def problem_p03338():
    n, s = int(eval(input())), eval(input())

    tmp = []

    for i in range(1, n):

        cnt = 0

        for j in list(set(s[:i])):

            if j in list(set(s[i:])):

                cnt += 1

        tmp.append(cnt)

    print((max(tmp)))


problem_p03338()

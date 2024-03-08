def problem_p03140():
    import string

    n = int(eval(input()))

    a = eval(input())

    b = eval(input())

    c = eval(input())

    cnt = 0

    for x, y, z in zip(a, b, c):

        wl = [0] * 26

        wl[string.ascii_lowercase.index(x)] += 1

        wl[string.ascii_lowercase.index(y)] += 1

        wl[string.ascii_lowercase.index(z)] += 1

        cnt += 3 - max(wl)

    print(cnt)


problem_p03140()

def problem_p03607():
    n = int(eval(input()))

    a = list(int(eval(input())) for _ in range(n))

    d = {}

    for i in range(n):

        d[a[i]] = d.get(a[i], 0) + 1

    ans = 0

    for v in list(d.values()):

        if v % 2 == 1:

            ans += 1

    print(ans)


problem_p03607()

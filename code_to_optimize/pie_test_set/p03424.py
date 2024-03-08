def problem_p03424():
    n = int(eval(input()))

    s = list(map(str, input().split()))

    total = []

    for i in range(n):

        if s[i] not in total:

            total.append(s[i])

    print(("Three" if len(total) == 3 else "Four"))


problem_p03424()

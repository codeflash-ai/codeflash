def problem_p03894():
    n, q = list(map(int, input().split()))

    exist = set([1])

    cup = list(range(n + 2))

    now = 1

    exist.add(cup[now - 1])

    exist.add(cup[now + 1])

    for i in range(q):

        a, b = list(map(int, input().split()))

        if now == a:
            now = b

        elif now == b:
            now = a

        tmp = cup[a]

        cup[a] = cup[b]

        cup[b] = tmp

        exist.add(cup[now - 1])

        exist.add(cup[now + 1])

    exist = list(exist)

    ans = 0

    for i in range(len(exist)):

        if exist[i] != 0 and exist[i] != n + 1:
            ans += 1

    print(ans)


problem_p03894()

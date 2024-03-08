def problem_p03487():
    from collections import Counter

    N = int(eval(input()))

    a = [int(i) for i in input().split()]

    b = Counter(a)

    ans = 0

    for x in b:

        if b[x] > x:

            ans += b[x] - x

        elif b[x] < x:

            ans += b[x]

    print(ans)


problem_p03487()

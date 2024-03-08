def problem_p03658():
    n, k = list(map(int, input().split()))

    l = [int(i) for i in input().split()]

    l.sort(reverse=True)

    ans = 0

    for j in range(k):

        ans += l[j]

    print(ans)


problem_p03658()

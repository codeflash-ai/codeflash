def problem_p03958():
    import sys

    k, n = list(map(int, input().split()))

    a = list(map(int, input().split()))

    a = [[a[i], i] for i in range(n)]

    if n == 1:
        print((a[0][0] - 1))
        sys.exit()

    pre = -1

    ans = 0

    for _ in range(k):

        a.sort(reverse=True)

        if a[0][1] != pre:

            a[0][0] -= 1

            pre = a[0][1]

        elif a[1][0] > 0:

            a[1][0] -= 1

            pre = a[1][1]

        else:

            ans = a[0][0]

            break

    print(ans)


problem_p03958()

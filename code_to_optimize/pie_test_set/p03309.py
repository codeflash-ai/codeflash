def problem_p03309():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    aa = []

    for i in range(n):

        aa.append(a[i] - (i + 1))

    b = sorted(aa)[n // 2]

    ans = 0

    for i in range(n):

        ans += abs(a[i] - (b + i + 1))

    print(ans)


problem_p03309()

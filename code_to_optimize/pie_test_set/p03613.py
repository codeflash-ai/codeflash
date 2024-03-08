def problem_p03613():
    from collections import Counter

    N = int(eval(input()))

    A = list(map(int, input().split()))

    d = Counter(A)

    ans = 0

    for i in range(1, 10**5 + 1):

        tmp = d[i - 1] + d[i] + d[i + 1]

        ans = max(tmp, ans)

    print(ans)


problem_p03613()

def problem_p03193():
    n, h, w = list(map(int, input().split()))

    ans = 0

    for i in range(n):

        a, b = list(map(int, input().split()))

        if a >= h and b >= w:

            ans += 1

    print(ans)


problem_p03193()

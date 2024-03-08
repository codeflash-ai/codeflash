def problem_p02832():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    cnt = 1

    ans = 0

    for i in range(n):

        if a[i] != cnt:

            ans += 1

        else:

            cnt += 1

    if ans == n:

        print((-1))

    else:

        print(ans)


problem_p02832()

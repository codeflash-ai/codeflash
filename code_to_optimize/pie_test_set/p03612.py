def problem_p03612():
    N = int(eval(input()))

    p = [int(x) for x in input().split()]

    ans = 0

    for i in range(N):

        if i < N - 1 and p[i] == i + 1:

            p[i] = p[i + 1]

            p[i + 1] = i + 1

            ans += 1

        elif i == N - 1 and p[i] == i + 1:

            p[i] = p[i - 1]

            p[i - 1] = i + 1

            ans += 1

    print(ans)


problem_p03612()

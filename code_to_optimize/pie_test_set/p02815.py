def problem_p02815():
    MOD = 1000000007

    N = int(eval(input()))

    C = list(map(int, input().split()))

    C.sort(reverse=True)

    ans = 0

    if N > 1:

        for i in range(N):

            ans += C[i] * (2 + i) * pow(2, N - 2, MOD) * pow(2, N, MOD)

            ans %= MOD

    else:

        ans = 2 * C[0] % MOD

    print(ans)


problem_p02815()

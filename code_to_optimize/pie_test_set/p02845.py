def problem_p02845():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    MOD = 1_000_000_007

    cnt = [0, 0, 0]

    ans = 1

    for i in range(N):

        ans *= sum(c == A[i] for c in cnt)

        ans %= MOD

        for j in range(3):

            if cnt[j] == A[i]:

                cnt[j] += 1

                break

    print(ans)


problem_p02845()

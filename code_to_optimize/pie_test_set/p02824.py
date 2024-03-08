def problem_p02824():
    N, M, V, P = list(map(int, input().split()))

    A = sorted(list(map(int, input().split())))

    B = A[: N - P + 1]

    S = [0]

    for i in range(N - P + 1):

        S.append(S[-1] + B[i])

    ans = P - 1

    for i in range(N - P + 1):

        if B[i] + M - B[-1] < 0:

            continue

        if B[i] * (N - P - i) - (S[-1] - S[i + 1]) + M * (N - P) >= max(M * (V - P), 0):

            ans += 1

    print(ans)


problem_p02824()

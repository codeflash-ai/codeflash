def problem_p03776():
    from scipy.misc import comb

    N, A, B = list(map(int, input().split()))

    V = list(map(int, input().split()))

    V.sort(reverse=True)

    print((sum(V[:A]) / A))

    Ath_CNT = V.count(V[A - 1])

    V_than_Ath_CNT = len([v for v in V if v > V[A - 1]])

    if max(V) == V[A - 1]:

        ans = 0

        for k in range(A, B + 1):

            ans += comb(Ath_CNT, k, 1)

    else:

        ans = comb(Ath_CNT, A - V_than_Ath_CNT, 1)

    print(ans)


problem_p03776()

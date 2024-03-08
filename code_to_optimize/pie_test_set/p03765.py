def problem_p03765():
    S = list(eval(input()))

    T = list(eval(input()))

    # 前計算

    S_n = [0] * (len(S) + 1)

    # S_n[0] = 1 if S[0] == 'A' else 2

    for i in range(1, len(S) + 1):

        if S[i - 1] == "A":

            S_n[i] = S_n[i - 1] + 1

        else:

            S_n[i] = S_n[i - 1] + 2

    T_n = [0] * (len(T) + 1)

    # T_n[0] = 1 if T[0] == 'A' else 2

    for i in range(1, len(T) + 1):

        if T[i - 1] == "A":

            T_n[i] = T_n[i - 1] + 1

        else:

            T_n[i] = T_n[i - 1] + 2

    q = int(eval(input()))

    for _ in range(q):

        a, b, c, d = list(map(int, input().split()))

        # output

        S_A = S_n[b] - S_n[a - 1]

        T_A = T_n[d] - T_n[c - 1]

        delta = abs(S_A - T_A)

        if delta % 3 == 0:

            print("YES")

        else:

            print("NO")


problem_p03765()

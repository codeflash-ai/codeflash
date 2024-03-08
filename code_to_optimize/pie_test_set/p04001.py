def problem_p04001():
    def Base_10_to_n(X, n):

        X_dumy = X

        out = ""

        while X_dumy > 0:

            out = str(X_dumy % n) + out

            X_dumy = int(X_dumy / n)

        return out

    S = eval(input())

    N = len(S) - 1

    ans = 0

    for n in range(2**N):

        b = Base_10_to_n(n, 2).rjust(N, "0")

        tmp = []

        ind = 0

        for i in range(N):

            if b[i] == "1":

                tmp.append(int(S[ind : i + 1]))

                ind = i + 1

        tmp.append(int(S[ind:]))

        ans += sum(tmp)

    print(ans)


problem_p04001()

def problem_p02938():
    L, R = list(map(int, input().split()))

    MOD = 10**9 + 7

    l = "{:060b}".format(L)[::-1]

    r = "{:060b}".format(R)[::-1]

    memo = [[[[-1 for l in range(2)] for k in range(2)] for j in range(2)] for i in range(60)]

    def f(pos, flagX, flagY, flagZ):

        if pos == -1:

            return 1

        if memo[pos][flagX][flagY][flagZ] != -1:

            return memo[pos][flagX][flagY][flagZ]

        ret = 0

        # x 0, y 0

        if flagX or l[pos] == "0":

            ret += f(pos - 1, flagX, 1 if r[pos] == "1" else flagY, flagZ)

        # x 0, y 1

        if (flagX or l[pos] == "0") and (flagY or r[pos] == "1") and flagZ:

            ret += f(pos - 1, flagX, flagY, flagZ)

        # x 1, y 1

        if flagY or r[pos] == "1":

            ret += f(pos - 1, 1 if l[pos] == "0" else flagX, flagY, 1)

        ret %= MOD

        memo[pos][flagX][flagY][flagZ] = ret

        return ret

    ans = f(59, 0, 0, 0)

    print(ans)


problem_p02938()

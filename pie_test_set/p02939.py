def problem_p02939(input_data):
    S = eval(input_data)

    N = len(S)

    ans = 0

    pre = ""

    now = ""

    for i in range(N):

        if pre == now or now == "":

            now += S[i]

        if pre != now:

            ans += 1

            pre = now

            now = ""

    return ans

def problem_p02555():
    from math import factorial as fac

    S = int(eval(input()))

    ans = 0

    if S == 1 or S == 2:

        print((0))

    elif S == 3:

        print((1))

    else:

        for k in range(1, S):

            if S - 2 * k - 1 >= 0 and S - 3 * k >= 0:

                ans += fac(S - 2 * k - 1) // (fac(k - 1) * fac(S - 3 * k))

        print((ans % 1000000007))


problem_p02555()

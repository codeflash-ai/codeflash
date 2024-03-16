def problem_p02555(input_data):
    from math import factorial as fac

    S = int(eval(input_data))

    ans = 0

    if S == 1 or S == 2:

        return 0

    elif S == 3:

        return 1

    else:

        for k in range(1, S):

            if S - 2 * k - 1 >= 0 and S - 3 * k >= 0:

                ans += fac(S - 2 * k - 1) // (fac(k - 1) * fac(S - 3 * k))

        return ans % 1000000007

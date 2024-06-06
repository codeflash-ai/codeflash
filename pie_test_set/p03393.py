def problem_p03393(input_data):
    import string

    S = eval(input_data)

    if len(S) < 26:

        for c in string.ascii_lowercase:

            if c not in S:

                break

        return S + c

    else:

        ret = None

        for i in range(26):

            if max(S[25 - i :]) != S[25 - i]:

                mc = "z"

                for c in S[25 - i :]:

                    if c > S[25 - i]:

                        mc = min(c, mc)

                ret = S[: 25 - i] + mc

                break

        if ret is None:

            return -1

        else:

            return ret

def problem_p00017(input_data):
    def chg(s, n):

        res = ""

        for i in s:

            o = ord(i)

            if 97 <= o <= 122:

                if o + n <= 122:

                    res += chr(o + n)

                else:

                    res += chr(o + n - 26)

            else:

                res += i

        return res

    while True:

        try:

            s = eval(input_data)

            for i in range(25, -1, -1):

                c = chg(s, i)

                e = c.split()

                if "the" in e or "this" in e or "that" in e:

                    return c

                    break

        except:

            break

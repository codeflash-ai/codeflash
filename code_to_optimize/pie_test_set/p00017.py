def problem_p00017():
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

            s = eval(input())

            for i in range(25, -1, -1):

                c = chg(s, i)

                e = c.split()

                if "the" in e or "this" in e or "that" in e:

                    print(c)

                    break

        except:

            break


problem_p00017()

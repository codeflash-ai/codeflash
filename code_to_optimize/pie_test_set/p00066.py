def problem_p00066():
    def f(a):

        for x in ["o", "x"]:

            if a[0::4].count(x) == 3 or a[2:7:2].count(x) == 3:
                return x

            for i in range(3):

                if a[i * 3 : i * 3 + 3].count(x) == 3 or a[i::3].count(x) == 3:
                    return x

        return "d"

    while 1:

        try:
            a = list(eval(input()))

        except:
            break

        print((f(a)))


problem_p00066()

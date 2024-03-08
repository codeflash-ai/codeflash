def problem_p00043():
    NUM = "0123456789"

    def atama(x):

        return [e for e in set(x) if x.count(e) >= 2]

    def koutu(x, i):

        return x.count(i) >= 3

    def shuntu(x, i):

        return i < 8 and x.count(i) > 0 and x.count(i + 1) > 0 and x.count(i + 2) > 0

    def make(x):

        return [int(c) for c in x]

    def check(x):

        for e in set(x):

            if koutu(x, e):

                y = [NUM[e] * 3]

                x1 = x + []

                x1.remove(e)

                x1.remove(e)

                x1.remove(e)

                if len(x1) == 0:
                    return y

                else:

                    y1 = check(x1)

                    if len(y1) > 0:
                        return y + y1

            elif shuntu(x, e):

                x1 = x + []

                y = [NUM[e : e + 3]]

                x1.remove(e)

                x1.remove(e + 1)

                x1.remove(e + 2)

                if len(x1) == 0:
                    return y

                else:

                    y1 = check(x1)

                    if len(y1) > 0:
                        return y + y1

        return ""

    try:

        while True:

            s = input()

            y = []

            for e in NUM[1:]:

                if s.count(e) >= 4:
                    continue

                x = make(s + e)

                for e0 in atama(x):

                    s0 = [NUM[e0] * 2]

                    x1 = x + []

                    x1.remove(e0)

                    x1.remove(e0)

                    s0 += check(x1)

                    if len(s0) == 5:

                        y.append(e)

                        break

            if len(y) == 0:
                print(0)

            else:

                for e in y:

                    print(e, end=" ")

                print()

    except:

        pass


problem_p00043()

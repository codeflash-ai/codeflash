def problem_p02467(input_data):
    import math

    def factorize(p):

        while p % 2 == 0:

            p //= 2

            yield 2

        r = 3

        while r < int(math.sqrt(p) + 1):

            if p % r == 0:

                p //= r

                yield r

            else:

                r += 2

        if p != 1:

            yield p

    n = int(eval(input_data))

    l = factorize(n)

    return (str(n) + ":", *list(l))

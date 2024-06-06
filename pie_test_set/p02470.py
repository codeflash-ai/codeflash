def problem_p02470(input_data):
    f = lambda x: p * (x - 1) // x

    p = n = int(eval(input_data))

    d = 2

    while d * d <= n:

        if n % d == 0:

            p = f(d)

            while n % d == 0:
                n //= d

        d += 1

    if n > 1:
        p = f(n)

    return p

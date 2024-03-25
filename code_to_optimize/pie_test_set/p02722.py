def problem_p02722(input_data):
    def make_divisors(n):

        divisors = []

        for i in range(1, int(n**0.5) + 1):

            if n % i == 0:

                divisors.append(i)

                if i != n // i:

                    divisors.append(n // i)

        # divisors.sort()

        return divisors

    n = int(eval(input_data))

    i = 2

    cnt = 1

    check = 0

    # for i in range(2, n + 1):

    #     m = n

    #     while m % i == 0:

    #         m //= i

    #     m %= i

    #     if m == 1:

    #         return (i)

    #         cnt += 1

    # return (cnt)

    # return ()

    # return ()

    out = [n]

    while i < n:  # 割ってひく

        m = n

        if i - m / i > 2:

            break

        while m % i == 0:

            check = 1

            m //= i

        if m % i == 1 and check:

            out.append(i)

            cnt += 1

        i += 1

    i = 2

    while i * i <= n:  # 割ってひく

        m = n

        while m % i == 0:

            m //= i

        if m == 1:

            out.append(i)

        i += 1

    out += make_divisors(n - 1)

    return len(set(out)) - 1

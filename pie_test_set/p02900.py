def problem_p02900(input_data):
    from math import sqrt, ceil

    a, b = list(map(int, input_data.split()))

    c_a = set()

    c_b = set()

    for i in range(1, ceil(sqrt(a)) + 1):

        if a % i == 0:

            c_a.add(i)

            if i > 1:

                while a % i == 0:

                    a = a // i

    if a != 1:

        c_a.add(a)

    for j in range(1, ceil(sqrt(b)) + 1):

        if b % j == 0:

            c_b.add(j)

            if j > 1:

                while b % j == 0:

                    b = b // j

    if b != 1:

        c_b.add(b)

    ans = c_a.intersection(c_b)

    return len(ans)

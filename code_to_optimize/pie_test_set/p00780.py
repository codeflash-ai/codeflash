def problem_p00780():
    from itertools import takewhile

    primes = [0, 0] + [1] * (2**15)

    for i in range(2, 182):

        if primes[i]:

            for j in range(i * i, 2**15 + 2, i):

                primes[j] = 0

    while True:

        n = int(eval(input()))

        if n == 0:

            break

        prime_values = [i for i in range(len(primes[:n])) if primes[i]]

        print(
            (
                sum(
                    n - p1 in prime_values
                    for p1 in takewhile(lambda x: x < n // 2 + 1, prime_values)
                )
            )
        )


problem_p00780()

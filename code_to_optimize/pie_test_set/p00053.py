def problem_p00053():
    def isPrime(x):

        if x == 2:

            return True

        if x < 2 or x % 2 == 0:

            return False

        i, root_x = 3, int(pow(x, 0.5))

        while i <= root_x:

            if x % i == 0:

                return False

            i += 2

        return True

    primes = [2]

    for i in range(3, 104730):

        if isPrime(i):

            primes.append(primes[-1] + i)

    while True:

        n = int(eval(input()))

        if n == 0:

            break

        print((primes[n - 1]))


problem_p00053()

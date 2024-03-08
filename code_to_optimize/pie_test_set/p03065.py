def problem_p03065():
    def gcd(a, b):

        while b:

            a, b = b, a % b

        return a

    def prime_decomposition(n):

        i = 2

        d = {}

        while i * i <= n:

            while n % i == 0:

                n //= i

                if i not in d:

                    d[i] = 0

                d[i] += 1

            i += 1

        if n > 1:

            if n not in d:

                d[n] = 1

        return d

    def eratosthenes(n):

        if n < 2:

            return []

        prime = []

        limit = n**0.5

        numbers = [i for i in range(2, n + 1)]

        while True:

            p = numbers[0]

            if limit <= p:

                return prime + numbers

            prime.append(p)

            numbers = [i for i in numbers if i % p != 0]

        return prime

    def ok(p):

        if A[0] % p != 0:

            return False

        B = [A[i] % p for i in range(1, N + 1)]

        mod = [0] * (p - 1)

        for i in range(N):

            mod[i % (p - 1)] += B[i]

            mod[i % (p - 1)] %= p

        return sum(mod) == 0

    N = int(eval(input()))

    A = [int(eval(input())) for i in range(N + 1)][::-1]

    g = abs(A[0])

    for a in A:

        g = gcd(g, abs(a))

    d = prime_decomposition(g)

    ans = [p for p in d]

    prime = eratosthenes(N + 1)

    for p in prime:

        if ok(p):

            ans.append(p)

    ans = list(set(ans))

    ans.sort()

    for p in ans:

        print(p)


problem_p03065()

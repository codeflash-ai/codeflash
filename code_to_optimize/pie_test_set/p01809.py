def problem_p01809():
    import fractions

    a, b = list(map(int, input().split()))

    b //= fractions.gcd(a, b)

    a, c = 2, 1

    while a**2 <= b:

        if b % a == 0:

            c *= a

            while b % a == 0:
                b //= a

        a += 1

    print((c * b))


problem_p01809()

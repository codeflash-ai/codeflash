def problem_p00005():
    import fractions

    while True:

        try:

            x, y = list(map(int, input().split()))

            print("%d %d" % (fractions.gcd(x, y), x / fractions.gcd(x, y) * y))

        except EOFError:

            break


problem_p00005()

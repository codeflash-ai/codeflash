def problem_p03307():
    import fractions

    def lcm(x, y):

        return (x * y) // fractions.gcd(x, y)

    N = int(eval(input()))

    print((lcm(N, 2)))


problem_p03307()

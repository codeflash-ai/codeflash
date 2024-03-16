def problem_p03307(input_data):
    import fractions

    def lcm(x, y):

        return (x * y) // fractions.gcd(x, y)

    N = int(eval(input_data))

    return lcm(N, 2)

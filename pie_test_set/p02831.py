def problem_p02831(input_data):
    import fractions

    a, b = list(map(int, input_data.split()))

    return a * b // fractions.gcd(a, b)

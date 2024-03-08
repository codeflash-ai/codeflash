def problem_p02831():
    import fractions

    a, b = list(map(int, input().split()))

    print((a * b // fractions.gcd(a, b)))


problem_p02831()

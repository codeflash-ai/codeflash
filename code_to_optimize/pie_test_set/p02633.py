def problem_p02633():
    from math import gcd

    x = int(eval(input()))

    y = x * 360 // gcd(x, 360)

    print((y // x))


problem_p02633()

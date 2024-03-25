def problem_p02633(input_data):
    from math import gcd

    x = int(eval(input_data))

    y = x * 360 // gcd(x, 360)

    return y // x

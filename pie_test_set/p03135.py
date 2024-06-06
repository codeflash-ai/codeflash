def problem_p03135(input_data):
    from decimal import Decimal

    t, x = input_data.split()

    return Decimal(t) / Decimal(x)

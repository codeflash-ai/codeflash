def problem_p02743(input_data):
    from decimal import Decimal

    a, b, c = list(map(int, input_data.split()))

    if Decimal(a).sqrt() + Decimal(b).sqrt() < Decimal(c).sqrt():

        return "Yes"

    else:

        return "No"

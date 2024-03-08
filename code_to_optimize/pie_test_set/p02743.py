def problem_p02743():
    from decimal import Decimal

    a, b, c = list(map(int, input().split()))

    if Decimal(a).sqrt() + Decimal(b).sqrt() < Decimal(c).sqrt():

        print("Yes")

    else:

        print("No")


problem_p02743()

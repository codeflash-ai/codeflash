def problem_p03964():
    import math

    from decimal import Decimal

    n = int(eval(input()))

    i = 0

    list = []

    while i < n:

        list.append(input().split())

        i += 1

    h = 0

    th = 1

    ah = 1

    while h < n:

        listh = list[h]

        x = int(listh[0])

        y = int(listh[1])

        m = max(math.ceil(th / x), math.ceil(ah / y))

        th = Decimal(m * x)

        ah = Decimal(m * y)

        h += 1

    print((th + ah))


problem_p03964()

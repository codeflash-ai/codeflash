def problem_p02755():
    # -*- coding: utf-8 -*-

    """

    Spyderエディタ



    これは一時的なスクリプトファイルです。

    """

    import math

    a, b = list(map(int, input().split()))

    alpha = []

    beta = []

    for x in range(math.floor(a / 0.08), math.floor(a / 0.05)):

        if math.floor(x * 0.08) == a:

            alpha.append(x)

    for y in range(math.floor(b / 0.10), math.floor(b / 0.05)):

        if math.floor(y * 0.1) == b:

            beta.append(y)

    if min(alpha) in beta:

        print((min(alpha)))

    elif min(beta) in alpha:

        print((min(beta)))

    else:

        print("-1")


problem_p02755()

def problem_p02755(input_data):
    # -*- coding: utf-8 -*-

    """

    Spyderエディタ



    これは一時的なスクリプトファイルです。

    """

    import math

    a, b = list(map(int, input_data.split()))

    alpha = []

    beta = []

    for x in range(math.floor(a / 0.08), math.floor(a / 0.05)):

        if math.floor(x * 0.08) == a:

            alpha.append(x)

    for y in range(math.floor(b / 0.10), math.floor(b / 0.05)):

        if math.floor(y * 0.1) == b:

            beta.append(y)

    if min(alpha) in beta:

        return min(alpha)

    elif min(beta) in alpha:

        return min(beta)

    else:

        return "-1"

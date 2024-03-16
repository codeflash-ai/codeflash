def problem_p02553(input_data):
    import numpy as np

    a, b, c, d = list(map(int, input_data.split()))

    hoge = []

    hoge.append(a * c)

    hoge.append(a * d)

    hoge.append(b * c)

    hoge.append(b * d)

    if max(hoge) < 0:

        if np.sign(a) != np.sign(b) or np.sign(c) != np.sign(d):

            return 0

        else:

            return max(hoge)

    else:

        return max(hoge)

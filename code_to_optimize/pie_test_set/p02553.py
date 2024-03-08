def problem_p02553():
    import numpy as np

    a, b, c, d = list(map(int, input().split()))

    hoge = []

    hoge.append(a * c)

    hoge.append(a * d)

    hoge.append(b * c)

    hoge.append(b * d)

    if max(hoge) < 0:

        if np.sign(a) != np.sign(b) or np.sign(c) != np.sign(d):

            print((0))

        else:

            print((max(hoge)))

    else:

        print((max(hoge)))


problem_p02553()

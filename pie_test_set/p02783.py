def problem_p02783(input_data):
    import collections

    import itertools as it

    import math

    # import numpy as np

    #  = input_data

    #  = int(input_data)

    h, a = list(map(int, input_data.split()))

    #  = list(map(int, input_data.split()))

    #  = [int(input_data) for i in range(N)]

    #

    # c = collections.Counter()

    if h % a == 0:

        return int(h / a)

    else:

        return h // a + 1

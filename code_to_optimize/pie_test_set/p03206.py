def problem_p03206(input_data):
    # encoding:utf-8

    import copy

    import numpy as np

    import random

    d = int(eval(input_data))

    christmas = "Christmas"

    plus = " Eve"

    return christmas + plus * (25 - d)

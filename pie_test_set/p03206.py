def problem_p03206(input_data):
    # encoding:utf-8

    import copy
    import random

    import numpy as np

    d = int(eval(input_data))

    christmas = "Christmas"

    plus = " Eve"

    return christmas + plus * (25 - d)

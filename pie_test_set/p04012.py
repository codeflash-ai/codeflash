def problem_p04012(input_data):
    import numpy as np

    w = eval(input_data)

    anal = np.array([])

    for i in range(ord("a"), ord("z") + 1):

        anal = np.append(anal, w.count(chr(i)))

    if np.all(anal % 2 == 0):

        # ans = np.sum(anal)

        return "Yes"

    else:

        return "No"

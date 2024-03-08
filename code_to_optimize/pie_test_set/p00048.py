def problem_p00048():
    import sys

    c = [
        "light fly",
        "fly",
        "bantam",
        "feather",
        "light",
        "light welter",
        "welter",
        "light middle",
        "middle",
        "light heavy",
        "heavy",
    ]

    w = [48, 51, 54, 57, 60, 64, 69, 75, 81, 91]

    for s in map(float, sys.stdin):

        i = 0

        while i < len(w) and w[i] < s:

            i += 1

        print(c[i])


problem_p00048()

def problem_p00075():
    import sys

    for s in sys.stdin:

        n, w, h = list(map(float, s.split(",")))

        bmi = w / h / h

        if bmi >= 25:
            print(int(n))


problem_p00075()

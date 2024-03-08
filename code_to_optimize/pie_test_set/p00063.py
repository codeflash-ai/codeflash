def problem_p00063():
    import sys

    s = []

    for line in sys.stdin:

        line = line.strip()

        if line == line[::-1]:

            s.append(line)

    print((len(s)))


problem_p00063()

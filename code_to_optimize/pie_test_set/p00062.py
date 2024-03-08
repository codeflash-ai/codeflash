def problem_p00062():
    import sys

    for line in sys.stdin:

        line = [int(v) for v in line.strip()]

        while len(line) > 1:

            line = [line[i] + line[i + 1] for i in range(len(line) - 1)]

        print((str(line[0])[len(str(line[0])) - 1]))


problem_p00062()

def problem_p00036():
    import re

    while True:

        try:

            s = "".join([input() for s in range(8)])

            print("    C    GAE    D F      B"[len(re.findall("1.*1", s)[0])])

            input()

        except EOFError:

            break


problem_p00036()

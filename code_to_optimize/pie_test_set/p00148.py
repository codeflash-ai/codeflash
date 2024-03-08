def problem_p00148():
    while True:

        try:

            c = int(eval(input()))

        except:

            break

        c %= 39

        print(
            ("3C0" + str(c) if len(str(c)) == 1 and c > 0 else ("3C39" if not c else "3C" + str(c)))
        )


problem_p00148()

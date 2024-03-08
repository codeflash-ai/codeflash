def problem_p03289():
    import string

    li = list(string.ascii_lowercase)

    s = list(eval(input()))

    if s[0] == "A" and s[2:-1].count("C") == 1:

        s.remove("C")

        s.remove("A")

    else:

        print("WA")

        exit()

    if all([s[i] in li for i in range(len(s))]):

        print("AC")

        exit()

    else:

        print("WA")


problem_p03289()

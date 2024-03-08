def problem_p03042():
    S = eval(input())

    a = S[:2]

    b = S[2:]

    if 1 <= int(a) <= 12 and 1 <= int(b) <= 12:

        print("AMBIGUOUS")

    elif 0 <= int(a) <= 99 and 1 <= int(b) <= 12:

        print("YYMM")

    elif 1 <= int(a) <= 12 and 0 <= int(b) <= 99:

        print("MMYY")

    else:

        print("NA")


problem_p03042()

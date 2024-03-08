def problem_p00110():
    import sys, re

    for e in sys.stdin:

        s = any([len(x) > 1 and x[0] == "X" for x in re.split("[+=]", e.strip())])

        for i in "0123456789"[s:]:

            if eval(e.replace("X", i).replace("=", "==")):
                print(i)
                break

        else:
            print("NA")


problem_p00110()

def problem_p03486():
    s = sorted(eval(input()))

    t = sorted(eval(input()))

    t.reverse()

    st = [s, t]

    if st == sorted(st) and s != t:

        print("Yes")

    else:

        print("No")


problem_p03486()

def problem_p03543():
    n = list(map(int, eval(input())))

    if n[0] == n[1] == n[2] or n[1] == n[2] == n[3]:

        print("Yes")

    else:

        print("No")


problem_p03543()

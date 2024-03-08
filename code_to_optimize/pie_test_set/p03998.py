def problem_p03998():
    SA = eval(input())

    SB = eval(input())

    SC = eval(input())

    pointer = SA[0]

    SA = SA[1:]

    d = {"a": SA, "b": SB, "c": SC}

    while True:

        if d[pointer] == "":

            ans = pointer

            break

        tmp = d[pointer][0]

        d[pointer] = d[pointer][1:]

        pointer = tmp

    print((ans.upper()))


problem_p03998()

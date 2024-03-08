def problem_p03369():
    s = eval(input())

    count = 0

    if s[0] == "o":

        count += 1

    if s[1] == "o":

        count += 1

    if s[2] == "o":

        count += 1

    print((700 + 100 * count))


problem_p03369()

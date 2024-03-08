def problem_p03986():
    s = "".join(input().split("ST"))

    cnts = 0

    cnt = 0

    for i in s:

        if i == "S":

            cnts += 1

        elif cnts > 0:

            cnts -= 1

            cnt += 1

    print((len(s) - cnt * 2))


problem_p03986()

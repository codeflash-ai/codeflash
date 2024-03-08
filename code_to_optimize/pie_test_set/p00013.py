def problem_p00013():
    import sys

    lst = []

    for line in sys.stdin:

        num = line.strip()

        if num == "0":

            print(lst.pop())

        else:

            lst.append(num)


problem_p00013()

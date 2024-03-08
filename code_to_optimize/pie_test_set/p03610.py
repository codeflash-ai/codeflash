def problem_p03610():
    s = eval(input())

    odd_string = ""

    for i, c in enumerate(s):

        if i % 2 == 0:

            odd_string += c

    print(odd_string)


problem_p03610()

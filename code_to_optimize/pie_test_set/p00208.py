def problem_p00208():
    while 4:

        n = int(eval(input()))

        if n == 0:
            break

        print((oct(n)[2:].translate(str.maketrans("4567", "5789"))))


problem_p00208()

def problem_p00175():
    while 1:

        a = eval(input())

        if a < 0:
            break

        x = []

        while a:

            x.append(str(a % 4))

            a /= 4

        if x == []:
            x = "0"

        print("".join(x[::-1]))


problem_p00175()

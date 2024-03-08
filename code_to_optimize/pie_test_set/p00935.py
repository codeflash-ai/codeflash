def problem_p00935():
    j = "".join

    n = int(eval(input()))

    d = j(j(input().split()) for i in [0] * (n // 19 + (n % 19 != 0)))

    i = 0

    while 1:

        if d.find(str(i)) < 0:

            print(i)

            exit()

        i += 1


problem_p00935()

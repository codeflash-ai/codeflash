def problem_p02420():
    while True:

        a = eval(input())

        if a == "-":

            break

        else:

            n = int(eval(input()))

            for i in range(n):

                h = int(eval(input()))

                a = a[h:] + a[:h]

        print(a)


problem_p02420()

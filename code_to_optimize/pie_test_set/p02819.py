def problem_p02819():
    x = int(eval(input()))

    f = 0

    if x == 2:

        print((2))

    else:

        for i in range(x, 10**6):

            for j in range(2, i // 2 + 2):

                if i % j == 0:

                    break

            else:

                f = i

                break

        print(f)


problem_p02819()

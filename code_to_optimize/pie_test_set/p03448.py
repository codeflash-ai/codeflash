def problem_p03448():
    max_n500 = int(eval(input()))

    max_n100 = int(eval(input()))

    max_n50 = int(eval(input()))

    amount = int(eval(input()))

    count = 0

    for n500 in range(max_n500 + 1):

        for n100 in range(max_n100 + 1):

            for n50 in range(max_n50 + 1):

                if n500 * 500 + n100 * 100 + n50 * 50 == amount:

                    count += 1

    print(count)


problem_p03448()

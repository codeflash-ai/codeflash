def problem_p03416():
    counter = 0

    c = list(map(int, input().split(" ")))

    for i in range(c[0], c[1] + 1):

        if str(i)[0] != str(i)[4]:

            continue

        if str(i)[1] != str(i)[3]:

            continue

        counter += 1

    print(counter)


problem_p03416()

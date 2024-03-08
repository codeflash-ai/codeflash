def problem_p02694():
    x = int(eval(input()))

    t = 100

    count = 0

    while True:

        if t >= x:

            print(count)

            exit()

        t = int(t + t * 0.01)

        count += 1


problem_p02694()

def problem_p02694(input_data):
    x = int(eval(input_data))

    t = 100

    count = 0

    while True:

        if t >= x:

            return count

            exit()

        t = int(t + t * 0.01)

        count += 1

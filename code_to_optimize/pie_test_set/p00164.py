def problem_p00164():
    from itertools import cycle

    INITIALIZE = 32

    while True:

        input_count = int(eval(input()))

        if input_count == 0:

            break

        get_list = [int(item) for item in input().split(" ")]

        ohajiki = INITIALIZE

        output = []

        for get_count in cycle(get_list):

            ohajiki -= (ohajiki - 1) % 5

            output.append(ohajiki)

            if get_count < ohajiki:

                ohajiki -= get_count

                output.append(ohajiki)

            else:

                output.append(0)

                break

        output = [str(item) for item in output]

        print(("\n".join(output)))


problem_p00164()

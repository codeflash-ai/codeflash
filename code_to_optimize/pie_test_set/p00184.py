def problem_p00184():
    judge = [10, 20, 30, 40, 50, 60, 121]

    while True:

        input_count = int(eval(input()))

        if input_count == 0:

            break

        data = [int(eval(input())) for _ in range(input_count)]

        counter = [0] * 7

        for item in data:

            for index, j in enumerate(judge):

                if item < j:

                    counter[index] += 1

                    break

        counter = [str(item) for item in counter]

        print(("\n".join(counter)))


problem_p00184()

def problem_p00162():
    MAX = 1000000

    hamming_list = [False] * (MAX + 1)

    hamming_list[0] = False

    hamming_list[1] = True

    for index in range(2, MAX + 1):

        if index / 2 % 1 == 0:

            if hamming_list[index // 2]:

                hamming_list[index] = True

        elif index / 3 % 1 == 0:

            if hamming_list[index // 3]:

                hamming_list[index] = True

        elif index / 5 % 1 == 0:

            if hamming_list[index // 5]:

                hamming_list[index] = True

    while True:

        input_data = eval(input())

        if input_data == "0":

            break

        start, end = [int(item) for item in input_data.split(" ")]

        count = sum(hamming_list[start : end + 1])

        print(count)


problem_p00162()

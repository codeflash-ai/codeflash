def problem_p02952(input_data):
    n = int(eval(input_data))

    def countKeta(num):

        count = 1

        while num / 10 >= 1:

            count += 1

            num = num // 10

        return count

    count = 0

    for i in range(1, n + 1):

        if countKeta(i) % 2 == 1:

            count += 1

    return count

def problem_p00491():
    # f = open("input.txt")

    # N, K = [int(x) for x in f.readline().split(' ')]

    # lines = f.readlines()

    # f.close()

    import sys

    N, K = [int(x) for x in sys.stdin.readline().split(" ")]

    lines = sys.stdin.readlines()

    schedule = [0] * N

    for line in lines:

        strs = line.split(" ")

        schedule[int(strs[0]) - 1] = int(strs[1])

    number = [0] * 9

    for i in range(9):

        l1 = i // 3

        l2 = i % 3

        if schedule[0] != l2 + 1 and schedule[0] != 0:

            number[i] = 0

        elif schedule[1] != l1 + 1 and schedule[1] != 0:

            number[i] = 0

        else:

            number[i] = 1

    for s in schedule[2:]:

        new_number = [0] * 9

        for i in range(9):

            l1 = i // 3

            l2 = i % 3

            for j in range(3):

                new_number[i] += (
                    number[l2 * 3 + j]
                    if ((s == 0 or s == l1 + 1) and not (l2 == j and l1 == j))
                    else 0
                )

        for i in range(9):

            number[i] = new_number[i] % 10000

    sum = 0

    for n in number:

        sum += n

    print((sum % 10000))


problem_p00491()

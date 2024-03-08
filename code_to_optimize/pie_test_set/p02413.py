def problem_p02413():
    r, c = map(int, input().split())

    table = []

    for _ in range(r):

        table.append(list(map(int, input().split())))

    for i in range(r):

        print(" ".join(map(str, table[i])), end="")

        print(" {}".format(sum(table[i])))

    total = 0

    for j in range(c):

        column_sum = 0

        column_sum = sum([row[j] for row in table])

        print("{} ".format(column_sum), end="")

        total += column_sum

    print(total)


problem_p02413()

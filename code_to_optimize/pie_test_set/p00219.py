def problem_p00219():
    #!/usr/bin/env python3

    while True:

        n = int(input())

        data = [0] * 10

        if n == 0:

            break

        for _ in range(n):

            in_data = int(input())

            data[in_data] += 1

        for d in data:

            for _ in range(d):

                print("*", end="")

            if d == 0:

                print("-")

            else:

                print(end="\n")


problem_p00219()

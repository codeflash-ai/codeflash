def problem_p00169():
    while True:

        cards = list(map(int, input().split()))

        if cards == [0]:

            break

        ace = 0

        sum = 0

        for i in cards:

            if i > 10:

                sum += 10

            else:

                sum += i

                if i == 1:

                    ace += 1

        for i in range(ace):

            if sum + 10 <= 21:

                sum += 10

        print(sum if sum <= 21 else 0)


problem_p00169()

def problem_p03482(input_data):

    S = eval(input_data)

    l = len(S)

    if l % 2 == 0:

        count = 0

        s = S[int(l / 2)]

        for i in range(int(l / 2)):

            if S[int(l / 2) - 1 - i] != s or S[int(l / 2) + i] != s:

                break

            count += 1

        return int(l / 2) + count

    else:

        s = S[int(l / 2)]

        count = 0

        for i in range(int(l / 2)):

            if S[int(l / 2) - 1 - i] != s or S[int(l / 2) + 1 + i] != s:

                break

            count += 1

        return int(l / 2) + 1 + count

def problem_p02667(input_data):
    t = eval(input_data)

    before_is_zero = True

    keeped_ones = 0

    n0 = 0

    ones = [0] * int(1e5 + 1)  # (number of zeros before 1, is_odd)

    ones_index = 0

    for i, x in enumerate(t):

        if x == "0":

            n0 += 1

            before_is_zero = True

        elif x == "1":

            if before_is_zero:

                ones[ones_index] = (n0, (i + 1) % 2)

                ones_index += 1

                before_is_zero = False

            else:

                keeped_ones += 1

                ones_index -= 1

                before_is_zero = True

    if before_is_zero:

        if len(ones) > 0:

            ones = ones[:-1]

    ans = keeped_ones * n0

    for i in range(ones_index):

        n0_before, is_odd = ones[i]

        ans += (n0_before + is_odd) // 2 + (n0 - n0_before)

    ans += (keeped_ones + (ones_index + 1) // 2) * (keeped_ones + ones_index // 2 + 1)

    return ans

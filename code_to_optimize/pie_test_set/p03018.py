def problem_p03018(input_data):
    s = eval(input_data)

    stock_bc = 0

    state = 0

    ans = 0

    for i in range(len(s)):

        if s[-1 - i] == "C":

            if state == 1:

                stock_bc = 0

            state = 1

        elif s[-1 - i] == "B":

            if state == 0:

                stock_bc = 0

            elif state == 1:

                stock_bc += 1

            state = 0

        elif s[-1 - i] == "A":

            if state == 0:

                ans += stock_bc

            else:

                stock_bc = 0

                state = 2

    return ans

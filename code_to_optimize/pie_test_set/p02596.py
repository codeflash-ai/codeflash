def problem_p02596(input_data):
    def c_repsept():

        K = int(eval(input_data))

        L = 9 * K // 7 if K % 7 == 0 else 9 * K

        if L % 2 == 0 or L % 5 == 0:

            return -1

        remainder = 1

        for n in range(1, L + 1):

            remainder = (10 * remainder) % L

            if remainder == 1:

                return n

        return -1

    return c_repsept()

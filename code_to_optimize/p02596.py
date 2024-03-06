def c_repsept():

    K = int(eval(input()))



    L = 9 * K // 7 if K % 7 == 0 else 9 * K

    if L % 2 == 0 or L % 5 == 0:

        return -1



    remainder = 1

    for n in range(1, L + 1):

        remainder = (10 * remainder) % L

        if remainder == 1:

            return n

    return -1



print((c_repsept()))
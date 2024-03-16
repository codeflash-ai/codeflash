def problem_p03943(input_data):
    # import time

    # starttime=time.clock()

    A, B, C = input_data.split()

    a = int(A)

    b = int(B)

    c = int(C)

    # return (a+c)

    if a > b:

        temp = a

        a = b

        b = temp

    if b > c:

        temp = b

        b = c

        c = temp

    if (a + b) == c:

        return "Yes"

    #     endtime = time.clock()

    #     return (endtime-starttime)

    else:

        return "No"

    #     endtime = time.clock()

    #     return (endtime-starttime)

    #

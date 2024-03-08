def problem_p03943():
    # import time

    # starttime=time.clock()

    A, B, C = input().split()

    a = int(A)

    b = int(B)

    c = int(C)

    # print(a+c)

    if a > b:

        temp = a

        a = b

        b = temp

    if b > c:

        temp = b

        b = c

        c = temp

    if (a + b) == c:

        print("Yes")

    #     endtime = time.clock()

    #     print(endtime-starttime)

    else:

        print("No")

    #     endtime = time.clock()

    #     print(endtime-starttime)

    #


problem_p03943()

def problem_p03331(input_data):
    n = int(eval(input_data))

    mn = float("inf")

    for i in range(1, n):

        count = 0

        for s in str(i):

            count += int(s)

        for p in str(n - i):

            count += int(p)

        if count < mn:

            mn = count

    return mn

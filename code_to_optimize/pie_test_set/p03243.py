def problem_p03243():
    num = str(eval(input()))

    a = []

    x = []

    for i in str(num):

        a.append(i)

    if a[1] <= a[0] and a[2] < a[0]:

        x.append(a[0])

        x.append(a[0])

        x.append(a[0])

        print(("".join(str(i) for i in x)))

    elif a[1] < a[0]:

        x.append(a[0])

        x.append(a[0])

        x.append(a[0])

        print(("".join(str(i) for i in x)))

    elif a[0] == a[1] == a[2]:

        x.append(a[0])

        x.append(a[0])

        x.append(a[0])

        print(("".join(str(i) for i in x)))

    else:

        z = int(a[0]) + 1

        x.append(z)

        x.append(z)

        x.append(z)

        print(("".join(str(i) for i in x)))


problem_p03243()

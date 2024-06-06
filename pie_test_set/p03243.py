def problem_p03243(input_data):
    num = str(eval(input_data))

    a = []

    x = []

    for i in str(num):

        a.append(i)

    if a[1] <= a[0] and a[2] < a[0]:

        x.append(a[0])

        x.append(a[0])

        x.append(a[0])

        return "".join(str(i) for i in x)

    elif a[1] < a[0]:

        x.append(a[0])

        x.append(a[0])

        x.append(a[0])

        return "".join(str(i) for i in x)

    elif a[0] == a[1] == a[2]:

        x.append(a[0])

        x.append(a[0])

        x.append(a[0])

        return "".join(str(i) for i in x)

    else:

        z = int(a[0]) + 1

        x.append(z)

        x.append(z)

        x.append(z)

        return "".join(str(i) for i in x)

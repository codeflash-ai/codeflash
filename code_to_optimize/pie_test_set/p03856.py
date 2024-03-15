def problem_p03856(input_data):
    s = input_data[::-1]

    d = ["maerd", "remaerd", "esare", "resare"]

    while s:

        for i in range(4):

            if s.startswith(d[i]):

                s = s[len(d[i]) :]

                break

        else:

            return "NO"

            exit()

    return "YES"

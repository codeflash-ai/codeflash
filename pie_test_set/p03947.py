def problem_p03947(input_data):
    s = input_data

    l = 0

    r = len(s) - 1

    news = ""

    i = 0

    while i < len(s):

        if s[i] == "W":

            while i < len(s) and s[i] == "W":

                i += 1

            news += s[i - 1]

        else:

            while i < len(s) and s[i] == "B":

                i += 1

            news += s[i - 1]

    return len(news) - 1

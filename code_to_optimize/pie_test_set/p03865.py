def problem_p03865(input_data):
    s = input_data.strip()

    is_first = True

    for s1, s2 in zip(s, s[2:]):

        if s1 != s2:

            is_first = False

    if is_first:

        return "Second"

    else:

        if len(s) % 2 == 0:

            if s[0] == s[-1]:

                return "First"

            else:

                return "Second"

        else:

            if s[0] == s[-1]:

                return "Second"

            else:

                return "First"

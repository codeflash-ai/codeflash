def problem_p03865():
    s = input().strip()

    is_first = True

    for s1, s2 in zip(s, s[2:]):

        if s1 != s2:

            is_first = False

    if is_first:

        print("Second")

    else:

        if len(s) % 2 == 0:

            if s[0] == s[-1]:

                print("First")

            else:

                print("Second")

        else:

            if s[0] == s[-1]:

                print("Second")

            else:

                print("First")


problem_p03865()

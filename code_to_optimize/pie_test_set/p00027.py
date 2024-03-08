def problem_p00027():
    from datetime import date

    while True:

        m, d = list(map(int, input().split()))

        if m == 0:

            break

        result = date(2004, m, d).isoweekday()

        print(
            (
                "Monday" * (result == 1)
                + "Tuesday" * (result == 2)
                + "Wednesday" * (result == 3)
                + "Thursday" * (result == 4)
                + "Friday" * (result == 5)
                + "Saturday" * (result == 6)
                + "Sunday" * (result == 7)
            )
        )


problem_p00027()

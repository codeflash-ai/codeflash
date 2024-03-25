def problem_p03129(input_data):
    n, k = list(map(int, input_data.split()))

    if n == 1 or n == 2:

        if k == 1:

            return "YES"

        else:

            return "NO"

    elif n % 2 == 0:

        cnt = n / 2

        if cnt >= k:

            return "YES"

        else:

            return "NO"

    else:

        cnt = n // 2 + 1

        if cnt >= k:

            return "YES"

        else:

            return "NO"

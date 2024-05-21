def problem_p03196(input_data):
    n, p = list(map(int, input_data.split()))

    if n == 1:

        return p

    elif n > 40:

        return 1

    else:

        ans = 1

        i = 1

        while True:

            if i**n > p:

                break

            if p % (i**n) == 0:

                ans = i

            i += 1

        return ans

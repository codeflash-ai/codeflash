def problem_p03316(input_data):
    # your code goes here

    n = int(eval(input_data))

    N = str(n)

    N = [int(x) for x in N]

    if n % sum(N) == 0:

        return "Yes"

    else:

        return "No"

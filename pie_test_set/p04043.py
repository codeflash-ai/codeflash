def problem_p04043(input_data):
    ABC = list(map(int, input_data.split()))

    if ABC.count(5) == 2 and ABC.count(7):

        return "YES"

    else:

        return "NO"

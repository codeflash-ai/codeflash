def problem_p03730(input_data):
    a, b, c = list(map(int, input_data.split()))

    for i in range(1, 100000):

        if a * i % b == c:

            return "YES"

            exit()

    return "NO"

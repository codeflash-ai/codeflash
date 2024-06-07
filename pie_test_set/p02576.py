def problem_p02576(input_data):
    array = list(map(int, input_data.split()))

    if array[0] % array[1] > 0:

        return (array[0] // array[1] + 1) * array[2]

    else:

        return (array[0] // array[1]) * array[2]

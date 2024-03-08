def problem_p02576():
    array = list(map(int, input().split()))

    if array[0] % array[1] > 0:

        print(((array[0] // array[1] + 1) * array[2]))

    else:

        print(((array[0] // array[1]) * array[2]))


problem_p02576()

def problem_p03840(input_data):
    i, o, t, j, l, s, z = list(map(int, input_data.split()))

    a = (i // 2 + j // 2 + l // 2) * 2

    return [a, max(a, ((i - 1) // 2 + (j - 1) // 2 + (l - 1) // 2) * 2 + 3)][
        (i and j and l) > 0
    ] + o

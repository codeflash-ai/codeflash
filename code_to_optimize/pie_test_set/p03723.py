def problem_p03723(input_data):
    a, b, c = list(map(int, input_data.split()))

    e = (a - b) | (b - c)

    return bool(e | (a | b | c) % 2) * (e ^ ~-e).bit_length() - 1

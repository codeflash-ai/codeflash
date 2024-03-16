def problem_p03671(input_data):
    ring = list(map(int, input_data.split()))

    return sum(ring) - max(ring)

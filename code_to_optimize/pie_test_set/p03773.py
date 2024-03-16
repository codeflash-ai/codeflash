def problem_p03773(input_data):
    a, b = list(map(int, input_data.split()))
    return (a + b) % 24

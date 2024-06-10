def problem_p03219(input_data):
    import sys

    input = sys.stdin.readline

    x, y = list(map(int, input_data.split()))

    return x + y // 2

def problem_p02612(input_data):
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    N = int(input_data)

    x = N + (-N) % 1000

    return x - N

def problem_p03480(input_data):
    #!/usr/bin/env python3

    # coding=utf-8

    import sys

    s = sys.stdin.readline().strip()

    l = [_s is "1" for _s in list(s)]

    l_r = l[::-1]

    index = 0

    for i, (c, n, c_r, n_r) in enumerate(
        zip(l[: len(s) // 2], l[1 : len(s) // 2 + 1], l_r[: len(s) // 2], l_r[1 : len(s) // 2 + 1])
    ):

        if c ^ n or c_r ^ n_r:

            index = i + 1

    return len(s) - index

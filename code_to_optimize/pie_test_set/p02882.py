def problem_p02882(input_data):
    from math import *

    def solve(B):

        α = b * sin((90 - B) * pi / 180) / sin(B * pi / 180)

        s = α * b / 2

        if α > a:
            s -= (α - a) * (b - b * a / α) / 2

        return s >= x / a

    a, b, x = list(map(int, input_data.split()))

    ok = 0

    ng = 90

    while ng - ok > 1e-7:

        mid = (ok + ng) / 2

        if solve(mid):
            ok = mid

        else:
            ng = mid

    return ok

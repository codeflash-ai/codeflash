def problem_p00874():
    from collections import Counter

    while True:

        W, D = list(map(int, input().split()))

        if not (W | D):

            break

        hw = [int(x) for x in input().split()]

        hd = [int(x) for x in input().split()]

        print(
            (sum(hw) + sum(hd) - sum(k * v for k, v in list((Counter(hw) & Counter(hd)).items())))
        )


problem_p00874()

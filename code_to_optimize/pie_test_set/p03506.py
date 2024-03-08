def problem_p03506():
    n, q, *t = list(map(int, open(0).read().split()))

    for v, w in zip(t[::2], t[1::2]):

        if ~-n:

            while v != w:

                if v > w:
                    v, w = w, v

                w = (w + n - 2) // n

        print(v)


problem_p03506()

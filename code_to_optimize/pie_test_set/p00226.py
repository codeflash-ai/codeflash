def problem_p00226():
    # AOJ 0226: Hit and Blow

    # Python3 2018.6.23 bal4u

    while 1:

        r, a = input().split()

        if r == "0":
            break

        h, sr, sa = 0, set(), set()

        for i in range(4):

            if r[i] == a[i]:
                h += 1

            else:

                sr.add(r[i])

                sa.add(a[i])

        print((h, len(sr & sa)))


problem_p00226()

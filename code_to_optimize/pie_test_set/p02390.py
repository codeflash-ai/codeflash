def problem_p02390():
    S = int(input())

    if S >= 3600:

        h = S // 3600

        m = S % 3600 // 60

        s = S % 60

    elif 3600 > S >= 60:

        h = 0

        m = S // 60

        s = S % 60

    else:

        h = m = 0

        s = S

    print(h, m, s, sep=":")


problem_p02390()

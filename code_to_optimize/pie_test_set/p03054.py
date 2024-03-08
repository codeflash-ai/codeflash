def problem_p03054():
    h, w, n = list(map(int, input().split()))

    sr, sc = list(map(int, input().split()))

    s = list(eval(input()))

    t = list(eval(input()))

    sl_move, sr_move, su_move, sd_move = [], [], [], []

    tl_move, tr_move, tu_move, td_move = [], [], [], []

    for i in s:

        if i == "L":

            sl_move += [-1]

            sr_move += [0]

            su_move += [0]

            sd_move += [0]

        elif i == "R":

            sl_move += [0]

            sr_move += [1]

            su_move += [0]

            sd_move += [0]

        elif i == "U":

            sl_move += [0]

            sr_move += [0]

            su_move += [-1]

            sd_move += [0]

        elif i == "D":

            sl_move += [0]

            sr_move += [0]

            su_move += [0]

            sd_move += [1]

    for i in t:

        if i == "L":

            tl_move += [-1]

            tr_move += [0]

            tu_move += [0]

            td_move += [0]

        elif i == "R":

            tl_move += [0]

            tr_move += [1]

            tu_move += [0]

            td_move += [0]

        elif i == "U":

            tl_move += [0]

            tr_move += [0]

            tu_move += [-1]

            td_move += [0]

        elif i == "D":

            tl_move += [0]

            tr_move += [0]

            tu_move += [0]

            td_move += [1]

    ru, rd, cr, cl = sr, sr, sc, sc

    for i in range(n):

        ru += su_move[i]

        if ru < 1:

            ans = "NO"

            break

        ru += td_move[i]

        if ru > h:

            ru = h

        rd += sd_move[i]

        if rd > h:

            ans = "NO"

            break

        rd += tu_move[i]

        if rd < 1:

            rd = 1

        cr += sr_move[i]

        if cr > w:

            ans = "NO"

            break

        cr += tl_move[i]

        if cr < 1:

            cr = 1

        cl += sl_move[i]

        if cl < 1:

            ans = "NO"

            break

        cl += tr_move[i]

        if cl > w:

            cl = w

        ans = "YES"

    print(ans)


problem_p03054()

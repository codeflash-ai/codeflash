def problem_p00082():
    cart = [4, 1, 4, 1, 2, 1, 2, 1]

    while True:

        try:

            que = list(map(int, input().split()))

            mx, mxcart = 0, "99999999"

            for sp in range(8):

                sm = 0

                for num in range(8):

                    if cart[(sp + num) % 8] <= que[num]:

                        sm += cart[(sp + num) % 8]

                    else:

                        sm += que[num]

                if sm > mx:

                    mx = sm

                    mxcart = "".join(map(str, cart[sp:] + cart[:sp]))

                elif sm == mx:

                    acart = "".join(map(str, cart[sp:] + cart[:sp]))

                    if int(mxcart) > int(acart):

                        mxcart = acart

            print(" ".join(map(str, mxcart)))

        except:

            break


problem_p00082()

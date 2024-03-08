def problem_p00446():
    while 1:

        t = eval(input())

        if t == 0:
            break

        card1 = []

        for i in range(t):

            card1 += [eval(input())]

        t *= 2

        card1 = sorted(card1)

        card2 = list(range(1, t + 1))

        for x in card1:

            card2.remove(x)

        ba = 0

        turn = 1

        while 1:

            if len(card1) == 0 or len(card2) == 0:
                break

            card = card1 if turn else card2

            cnt = 0

            while cnt < len(card):

                if card[cnt] > ba:
                    break

                cnt += 1

            if cnt != len(card):

                ba = card.pop(cnt)
                turn = 1 - turn

            else:

                ba = 0
                turn = 1 - turn

        print(len(card2))

        print(len(card1))


problem_p00446()

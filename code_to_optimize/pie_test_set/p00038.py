def problem_p00038():
    while True:

        try:

            hand = list(map(int, input().split(",")))

            kind = list(set(hand))

            rank = []

            for card in kind:

                rank.append(hand.count(card))

                rank.sort()

                rank.reverse()

            if rank[0] == 4:

                print("four card")

            elif rank[0] == 3:

                if rank[1] == 2:

                    print("full house")

                else:

                    print("three card")

            elif rank[0] == 2:

                if rank[1] == 2:

                    print("two pair")

                else:

                    print("one pair")

            else:

                hand.sort()

                if hand[4] - hand[0] == 4 or (hand[0] == 1 and hand[1] == 10):

                    print("straight")

                else:

                    print("null")

        except:

            break


problem_p00038()

def problem_p02408():
    n = int(input())

    cards, suits = [], ["S", "H", "C", "D"]

    for suit in suits:

        for x in range(1, 14):

            cards.append("{} {}".format(suit, x))

    for _ in range(n):

        target = " ".join(input().split())

        if target in cards:

            cards.remove(target)

    [print(c) for c in cards]


problem_p02408()

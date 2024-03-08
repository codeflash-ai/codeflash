def problem_p02261():
    def bubbleSort(cards):

        n = len(cards)

        for i in range(0, n):

            for j in range(n - 1, i, -1):

                if int(cards[j][1]) < int(cards[j - 1][1]):

                    cards[j], cards[j - 1] = cards[j - 1], cards[j]

        print((" ".join(map(str, cards))))

    def selectionSort(cards):

        n = len(cards)

        for i in range(0, n):

            mini = i

            for j in range(i, n):

                if int(cards[j][1]) < int(cards[mini][1]):

                    mini = j

            if mini != i:

                cards[i], cards[mini] = cards[mini], cards[i]

        print((" ".join(map(str, cards))))

    n = int(eval(input()))

    cards = input().split(" ")

    cards2 = list(cards)

    bubbleSort(cards)

    print("Stable")

    selectionSort(cards2)

    if " ".join(map(str, cards2)) == " ".join(map(str, cards)):

        print("Stable")

    else:

        print("Not stable")


problem_p02261()

def problem_p02385():
    class Dice:

        def __init__(self, label):

            self.label = label

        def north(self):

            self.change([2, 6, 3, 4, 1, 5])

        def west(self):

            self.change([3, 2, 6, 1, 5, 4])

        def east(self):

            self.change([4, 2, 1, 6, 5, 3])

        def south(self):

            self.change([5, 1, 3, 4, 6, 2])

        def change(self, convert):

            num = []

            for i in range(6):

                num.append(self.label[convert[i] - 1])

            self.label = num

    def main():

        f = [int(x) for x in input().split()]

        s = [int(x) for x in input().split()]

        dice = Dice(f)

        labels = []

        for i in range(6):

            if i < 4:

                dice.north()

            elif i == 4:

                dice.east()

                dice.south()

            elif i == 5:

                dice.south()

                dice.south()

            for j in range(4):

                dice.west()

                labels.append(dice.label)

        if s in labels:

            print("Yes")

        else:

            print("No")

    if __name__ == "__main__":

        main()


problem_p02385()

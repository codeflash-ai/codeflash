def problem_p00201():
    while True:

        things = {}

        count = eval(input())

        if count == 0:
            break

        for i in range(count):

            fuga = input().split()

            things[fuga[0]] = int(fuga[1])

        for i in range(eval(input())):

            fuga = input().split()

            money = 0

            for j in range(int(fuga[1])):

                hoge = things[fuga[j + 2]] if fuga[j + 2] in things else 0

                money += hoge

            things[fuga[0]] = things[fuga[0]] if money > things[fuga[0]] else money

        ans = input()

        fin = things[ans] if ans in things else 0

        print(fin)


problem_p00201()

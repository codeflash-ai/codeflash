def problem_p01322():
    import re

    while 1:

        n, m = list(map(int, input().split()))

        if n == 0:
            break

        prize = []

        for i in range(n):

            num, money = input().replace("*", "[0-9]").split()

            prize.append([re.compile(num), int(money)])

        ans = 0

        for i in range(m):

            lot = input()

            for num, money in prize:

                if num.search(lot):

                    ans += money

        print(ans)


problem_p01322()

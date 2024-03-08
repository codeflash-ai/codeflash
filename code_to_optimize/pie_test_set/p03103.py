def problem_p03103():
    # from collections import defaultdict

    # N,M = list(map(int,input().split()))

    # price_howmany = defaultdict(int)

    import sys

    stdin = sys.stdin

    read_int = lambda: list(map(int, stdin.readline().split()))

    N, M = read_int()

    price_howmany = {}

    # price_list = []

    price_set = set()

    for _ in range(N):

        tmp = read_int()

        # if tmp[0] not in price_set:

        # price_list.append(tmp[0])

        price_set.add(tmp[0])

        if tmp[0] in price_howmany:

            price_howmany[tmp[0]] += tmp[1]

        else:

            price_howmany[tmp[0]] = tmp[1]

    # price_list.sort(reverse=True)

    price_list = sorted(list(price_set))

    # price_howmany = dict(price_howmany)

    def solve():

        price = 0

        global M

        # while M > 0:

        # low_price = min(price_howmany.keys())

        # low_price = price_list.pop()

        # many = min(M,price_howmany.pop(low_price))

        # price += many * low_price

        # M -= many

        for low_price in price_list:

            many = min(M, price_howmany.pop(low_price))

            price += many * low_price

            M -= many

            if M == 0:
                break

        print(price)

    if __name__ == "__main__":

        solve()


problem_p03103()

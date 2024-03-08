def problem_p02603():
    from bisect import bisect_right

    def main():

        N = int(eval(input()))

        A = list(map(int, input().split()))

        INF = 10**18

        last_max = A[0]

        last_min = A[0]

        money = 1000

        stock = 0

        old_trend = 0

        trend = A[1] - A[0]

        buy_point = []

        bought_point = []

        for i in range(N - 1):

            trend = A[i + 1] - A[i]

            if trend * old_trend > 0 or trend == 0:

                continue

            else:

                if trend > 0:

                    buy_point.append(i)

                if trend < 0:

                    bought_point.append(i)

            old_trend = trend

        if len(buy_point) > len(bought_point):

            bought_point.append(N - 1)

        for i in range(N):

            if i in buy_point:

                buy = money // A[i]

                stock += buy

                money -= buy * A[i]

                nx_bought_idx = bisect_right(bought_point, i)

                bought_value = A[nx_bought_idx]

                j = i - 1

                while j >= 0 and money >= 100:

                    if A[j] < bought_value and money >= A[i]:

                        money -= A[j]

                        stock += 1

                        break

                    j -= 1

            if i in bought_point:

                bought = stock

                stock = 0

                money += bought * A[i]

        print(money)

    if __name__ == "__main__":

        main()


problem_p02603()

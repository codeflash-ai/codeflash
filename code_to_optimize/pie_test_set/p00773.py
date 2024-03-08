def problem_p00773():
    from time import time

    def change(before_tax, after_tax, previous_price):

        # change z from taxrate x to taxrate y

        original_price = 0

        for i in range(1, previous_price + 1):

            if i * (100 + before_tax) // 100 == previous_price:

                original_price = i

                break

            else:

                pass

        return original_price * (100 + after_tax) // 100

    # l = []

    while True:

        x, y, s = [int(x) for x in input().split()]

        if x == 0:

            break

        else:

            ans = 0

            for i in range(1, s):

                price1, price2 = i, s - i

                afterprice = change(x, y, price1) + change(x, y, price2)

                if afterprice > ans:

                    ans = afterprice

                else:

                    continue

            # l.append(ans)

            print(ans)

    """
    
    for x in l:
    
        print(x)
    
    """


problem_p00773()

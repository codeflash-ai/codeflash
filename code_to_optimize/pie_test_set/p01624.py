def problem_p01624():
    # undo が出来るゲームは全て2手だけ見ればよい

    ops = ["+", "*", "-", "&", "^", "|"]

    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

    def check(x):

        # +*

        op = 0

        for c in x:

            if c in ops:

                op += 1

                if op >= 2:

                    return None

            else:

                op = 0

        # +...

        for op in ops:

            if x.startswith(op):

                return None

            if ("(" + op) in x:

                return None

        # 0がないか

        zero_ok = False

        for c in x:

            if not zero_ok and c == "0":

                return None

            if c in ops:

                zero_ok = False

            elif c in numbers:

                zero_ok = True

            else:  # ( )

                zero_ok = False

        try:

            val = int(eval(x))

            return val

        except:

            return None

    def get_nexts(x):

        # 削除

        result = []

        for i in range(len(x)):

            y = x[:i] + x[i + 1 :]

            val = check(y)

            if val != None:

                result.append((val, y))

        # 追加

        for i in range(len(x) + 1):

            add_list = numbers + ops

            for s in add_list:

                y = x[:i] + s + x[i:]

                val = check(y)

                if val != None:

                    result.append((val, y))

        return result

    while True:

        n, x = input().split(" ")

        n = int(n)

        if n == 0:

            quit()

        nexts = get_nexts(x)

        if n == 1:

            nexts.sort(key=lambda a: -a[0])

            print((nexts[0][0]))

            continue

        maxval = eval(x)

        tele = x

        minvals = []

        for val, y in nexts:

            nextss = get_nexts(y)

            nextss.sort(key=lambda a: a[0])

            minvals.append(nextss[0][0])

            if maxval < nextss[0][0]:

                maxval = nextss[0][0]

                tele = nextss[0][1]

        if n % 2 == 0:

            print((max(minvals)))  # 999+9999 -> 9&99+9999

            continue

        nexts = get_nexts(tele)

        if n % 2 == 1:

            nexts.sort(key=lambda a: -a[0])

            print((nexts[0][0]))

            continue


problem_p01624()

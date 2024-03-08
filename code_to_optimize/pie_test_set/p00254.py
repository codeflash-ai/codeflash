def problem_p00254():
    while 1:

        n = list(input())

        if n == ["0"] * 4:
            break

        elif n[0] == n[1] == n[2] == n[3]:

            print("NA")

            continue

        count = 0

        while 1:

            if n == ["6", "1", "7", "4"]:

                print(count)

                break

            n = sorted(n)

            l = int("".join(map(str, n[::-1])))

            s = int("".join(map(str, n[n.count("0") :])))

            n = list(str(l - s))

            n = ["0"] * (4 - len(n)) + n

            count += 1


problem_p00254()

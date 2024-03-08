def problem_p02406():
    n = int(input())

    i = 1

    SP = []

    while i <= n:

        x = i

        if x % 3 == 0:

            SP.append(str(i))

        else:

            while x:

                if x % 10 == 3:

                    SP.append(str(i))

                    break

                x /= 10

        i += 1

    print("", " ".join(SP))


problem_p02406()

def problem_p00044():
    import sys

    import math

    primes = [2]

    for line in sys.stdin:

        try:

            n = int(line)

            for i in range(max(primes) + 1, n):

                if i % 2 == 0:

                    continue

                elif math.sqrt(i) == 0:

                    continue

                elif len([j for j in primes if i % j == 0]) > 0:

                    continue

                primes.append(i)

            if n == 3:

                print(2, end=" ")

            else:

                for i in range(n - 1, 1, -1):

                    if i % 2 == 0:

                        continue

                    elif math.sqrt(i) == 0:

                        continue

                    elif len([j for j in primes if i % j == 0 and i != j]) > 0:

                        continue

                    print(i, end=" ")

                    break

            for i in range(n + 1, n + 1000):

                if i % 2 == 0:

                    continue

                elif math.sqrt(i) == 0:

                    continue

                elif len([j for j in primes if i % j == 0 and i != j]) > 0:

                    continue

                print(i)

                break

        except:

            break


problem_p00044()

def problem_p04016():
    import sys

    input = sys.stdin.readline

    def digit_sum(n, b):

        res = 0

        while n > 0:

            res += n % b

            n //= b

        return res

    def main():

        n = int(eval(input()))

        s = int(eval(input()))

        ans = -1

        if n < s:

            ans = -1

        elif n == s:

            ans = n + 1

        else:

            key = []

            start, first = n, 1

            for i in range(1, n):

                first = digit_sum(n, start)

                end = start - n // i // (i + 1)

                if start == end + 1:

                    judge = start

                    break

                while end > 1:

                    if digit_sum(n, end) == i * (start - end) + first:

                        end -= 1

                        continue

                    break

                next_start = end

                value = digit_sum(n, start)

                end = value + (start - end - 1) * i

                key.append((start, value, end, i, first))

                judge = start

                start = next_start

                if start == 1:

                    break

            for i in range(2, judge + 1):

                sub = digit_sum(n, i)

                if sub == s:

                    ans = i

                    break

            else:

                for index, start, end, i, first in reversed(key):

                    if end < s or s < start:

                        continue

                    if (s - start) % i:

                        continue

                    ans = index - (s - first) // i

                    break

        print(ans)

    if __name__ == "__main__":

        main()


problem_p04016()

def problem_p02959():
    from sys import stdin, stdout

    if __name__ == "__main__":

        n = int(stdin.readline())

        a = [int(x) for x in stdin.readline().split()]

        b = [int(x) for x in stdin.readline().split()]

        cnt = 0

        for i in range(n):

            if a[i] >= b[i]:

                cnt += b[i]

            else:

                cnt += a[i]

                d = b[i] - a[i]

                cnt += min(a[i + 1], d)

                if d > a[i + 1]:

                    a[i + 1] = 0

                else:
                    a[i + 1] -= d

        stdout.write(str(cnt) + "\n")


problem_p02959()

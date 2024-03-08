def problem_p02631():
    import sys

    from collections import deque

    input = sys.stdin.readline

    def main():

        n = int(input())

        input_a = list(map(int, input().split()))

        a = deque(input_a)

        while len(a) != 1:

            x = a.popleft()

            y = a.popleft()

            z = x ^ y

            a.append(z)

        sum = a.popleft()

        ans = []

        for i in input_a:

            x = i ^ sum

            ans.append(x)

        for i in ans:

            print(i, end=" ")

    main()


problem_p02631()

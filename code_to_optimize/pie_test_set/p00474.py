def problem_p00474():
    def solve():

        n, l = list(map(int, input().split()))

        ans = 0

        pre = 0

        up_acc = 0

        down_acc = 0

        for i in range(n):

            length = int(eval(input()))

            time = l - length

            if length > pre:

                up_acc += time

                if down_acc > ans:

                    ans = down_acc

                down_acc = time

            else:

                down_acc += time

                if up_acc > ans:

                    ans = up_acc

                up_acc = time

            pre = length

        else:

            ans = max(ans, up_acc, down_acc)

        print(ans)

    solve()


problem_p00474()

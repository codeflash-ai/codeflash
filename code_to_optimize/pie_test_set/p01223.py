def problem_p01223():
    for i in range(eval(input())):

        n = eval(input())

        a = list(map(int, input().split()))

        l = [a[i + 1] - a[i] for i in range(n - 1)]

        print(max(0, max(l)), abs(min(0, min(l))))


problem_p01223()

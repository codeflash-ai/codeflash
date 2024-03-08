def problem_p00003():
    for n in range(eval(input())):

        a, b, c = sorted(map(int, input().split()))

        s = "YES" if a * a + b * b == c * c else "NO"

        print(s)


problem_p00003()

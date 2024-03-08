def problem_p02407():
    n = int(input())

    ls = list(map(int, input().split()))

    print(" ".join(map(str, ls[::-1])))


problem_p02407()

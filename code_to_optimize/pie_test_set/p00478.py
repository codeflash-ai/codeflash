def problem_p00478():
    s = eval(input())

    print((sum(1 for _ in range(int(eval(input()))) if s in 2 * eval(input()))))


problem_p00478()

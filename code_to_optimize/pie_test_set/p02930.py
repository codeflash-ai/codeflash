def problem_p02930():
    N, j = int(eval(input())), 0
    exec("print(*[9-format(i+1,'09b').rfind('1')for i in range(N)][:N-j-1]);j+=1;" * N)


problem_p02930()

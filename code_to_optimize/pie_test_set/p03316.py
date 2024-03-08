def problem_p03316():
    # your code goes here

    n = int(eval(input()))

    N = str(n)

    N = [int(x) for x in N]

    if n % sum(N) == 0:

        print("Yes")

    else:

        print("No")


problem_p03316()

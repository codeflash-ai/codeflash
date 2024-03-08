def problem_p03580():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    N = int(readline())

    S = readline().rstrip().decode("utf-8")

    def solve_partial(S):

        INF = 10**18

        """
    
        ・Sは1から始まり、1で終わる
    
        ・Sは00を含まない
    
        ・したがって、Sは1,01に分解可能
    
        ・残る最小個数を調べるdp。これは、1, 101111,111101 の3種を数えることと同じ
    
        ・a, b0cccc, dddd0e として、「現在の1がどれであるか -> 最小個数」でdp
    
        ・個数はa,b,eのときに数える
    
        """

        S = S.replace("01", "2")

        a, b, c, d, e = 1, 1, INF, 0, INF

        for x in S[1:]:

            if x == "1":

                a2 = min(a, c, e) + 1

                b2 = min(a, c, e) + 1

                c2 = c

                d2 = min(a, c, d, e)

                e2 = INF

            else:

                a2 = min(a, c, e) + 1

                b2 = min(a, c, e) + 1

                c2 = b

                d2 = min(a, c, e)

                e2 = d + 1

            a, b, c, d, e = a2, b2, c2, d2, e2

        return len(S) - min(a, c, e)

    answer = 0

    for x in S.split("00"):

        x = x.strip("0")

        if x:

            answer += solve_partial(x)

    print(answer)


problem_p03580()

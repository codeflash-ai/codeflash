def problem_p03220():
    import sys

    N = eval(input())

    TA = eval(input())

    T, A = TA.split()

    Hinp = eval(input())

    H = Hinp.split()

    ansnum = None

    ans = 0

    def getdiff(A, ans):

        return abs(max(ans, A) - min(ans, A))

    def chk(A, _ans, ans):

        new_d, old_d = getdiff(A, _ans), getdiff(A, ans)

        if new_d < old_d:

            return True

        return False

    for i, x in enumerate(H):

        _ans = int(T) - int(x) * 0.006

        if _ans == A:

            break

        if not ansnum or chk(int(A), _ans, ansnum):

            ans = i + 1

            ansnum = _ans

    print(ans)


problem_p03220()

def problem_p02742():
    ## coding: UTF-8

    from decimal import *

    s = input().split()

    t = [int(p) for p in s]

    # print(t)

    if t[0] == 1 or t[1] == 1:

        answer = 1

    elif Decimal(Decimal(t[0]) % 2) == 1 and Decimal(Decimal(t[1]) % 2) == 1:

        n = Decimal(Decimal(Decimal(t[0]) - 1) // 2)

        # print(n)

        m = Decimal(Decimal(Decimal(t[1]) - 1) // 2)

        answer = Decimal(
            Decimal(Decimal(Decimal(m) + 1) * Decimal(Decimal(n) + 1))
            + Decimal(Decimal(m) * Decimal(n))
        )

    else:

        answer = Decimal(Decimal(Decimal(t[0]) * Decimal(t[1])) // 2)

    print((int(answer)))

    """
    
    H = t[0]
    
    W = t[1]
    
    ans = 0
    
    if( W % 2 == 0):
    
        for i in range(H):
    
            ans += W / 2
    
    else:
    
        for i in range(H):
    
            if(i % 2 == 0):
    
                ans += W / 2.0 + 0.5
    
            else:
    
                ans += W / 2.0 - 0.5
    
    print(int(ans))
    
    """


problem_p02742()

def problem_p02354():
    N, S = list(map(int, input().split()))

    As = list(map(int, input().split()))

    i = 0

    j = 0

    sum = 0

    m_l = float("inf")

    while True:

        while j < N and sum < S:

            sum += As[j]

            j += 1

        if sum < S:

            break

        m_l = min(j - i, m_l)

        sum -= As[i]

        i += 1

    """
    
    while j < N:
    
    	sum += As[j]
    
    	j += 1
    
    	while i < j and sum :
    
    		sum -= As[i]
    
    		i += 1
    
    		if sum >= S:
    
    			continue
    
    		m_l = min(j-i+1, m_l)
    
    		break
    
    """

    print((m_l if m_l != float("inf") else 0))


problem_p02354()

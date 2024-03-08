def problem_p00190():
    from math import factorial

    from queue import PriorityQueue

    FACTORIAL = [factorial(i) for i in range(13)]

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    MOVE = [[0] for u in range(13)]

    MOVE[0] = [-1, -1, -1, 2]

    MOVE[1] = [-1, -1, 2, 5]

    MOVE[2] = [1, 0, 3, 6]

    MOVE[3] = [2, -1, -1, 7]

    MOVE[4] = [-1, -1, 5, -1]

    MOVE[5] = [4, 1, 6, 9]

    MOVE[6] = [5, 2, 7, 10]

    MOVE[7] = [6, 3, 8, 11]

    MOVE[8] = [7, -1, -1, -1]

    MOVE[9] = [-1, 5, 10, -1]

    MOVE[10] = [9, 6, 11, 12]

    MOVE[11] = [10, 7, -1, -1]

    MOVE[12] = [-1, 10, -1, -1]

    """
    
    MOVE[0][LEFT]=MOVE[1][LEFT]=MOVE[4][LEFT]=MOVE[9][LEFT]=MOVE[12][LEFT]=False
    
    MOVE[0][UP]=MOVE[1][UP]=MOVE[3][UP]=MOVE[4][UP]=MOVE[8][UP]=False
    
    MOVE[0][RIGHT]=MOVE[3][RIGHT]=MOVE[8][RIGHT]=MOVE[11][RIGHT]=MOVE[12][RIGHT]=False
    
    MOVE[4][DOWN]=MOVE[8][DOWN]=MOVE[9][DOWN]=MOVE[11][DOWN]=MOVE[12][DOWN]=False
    
    """

    def hash(cell):

        work = cell[:]

        hash = 0

        for i in range(12):

            hash += work[i] * FACTORIAL[13 - 1 - i]

            for ii in range(i + 1, 13):

                if work[ii] > work[i]:

                    work[ii] -= 1

        return hash

    def dehash(key):

        cell = []

        for i in range(13):

            cell.append(key / FACTORIAL[13 - 1 - i])

            key %= FACTORIAL[13 - 1 - i]

        for i in range(13 - 1, -1, -1):

            for ii in range(i + 1, 13):

                if cell[i] <= cell[ii]:

                    cell[ii] += 1

        return cell

    def evaluate(cell):

        point = [
            [0, 2],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [4, 2],
        ]

        eva = 0

        for i in range(0, 13):

            if not (cell[i] == 0 or cell[i] == 12):

                eva += abs(point[cell[i]][0] - point[i][0])

                eva += abs(point[cell[i]][1] - point[i][1])

        return eva

    ANS_HASH = [
        hash([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        hash([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]),
    ]

    while True:

        p = [eval(input())]

        if p == [-1]:

            break

        for u in range(4):

            for pp in map(int, input().split()):

                p.append(pp)

        p[p.index(0)] = 12

        pq = PriorityQueue()

        pq.put([evaluate(p), hash(p), 0])

        visited = {}

        visited[hash(p)] = True

        ans = 0 if hash(p) in ANS_HASH else "NA"

        # cur=[eva,hashkey,step]

        while not pq.empty():

            unused, cur_hash, cur_step = pq.get()

            cur_cell = dehash(cur_hash)

            """
    
            print "STEP",cur_step
    
            print [cur_cell[0]]
    
            print cur_cell[1:4]
    
            print cur_cell[4:9]
    
            print cur_cell[9:12]
    
            print [cur_cell[12]]
    
            print
    
            """

            if not (cur_step < 20 and ans == "NA"):

                break

            for i in range(13):

                if cur_cell[i] == 0 or cur_cell[i] == 12:

                    # print cur_cell[i]

                    for ii in [LEFT, UP, RIGHT, DOWN]:

                        # print ii,

                        if not MOVE[i][ii] == -1:

                            # print "MOVE"

                            cur_cell[i], cur_cell[MOVE[i][ii]] = cur_cell[MOVE[i][ii]], cur_cell[i]

                            """
    
                            print "MOVING",ii
    
                            print [cur_cell[0]]
    
                            print cur_cell[1:4]
    
                            print cur_cell[4:9]
    
                            print cur_cell[9:12]
    
                            print [cur_cell[12]]
    
                            """

                            hashkey = hash(cur_cell)

                            if not hashkey in visited:

                                if hashkey in ANS_HASH:

                                    ans = cur_step + 1

                                    break

                                pq.put([evaluate(cur_cell) + cur_step + 1, hashkey, cur_step + 1])

                                visited[hashkey] = True

                            cur_cell[i], cur_cell[MOVE[i][ii]] = cur_cell[MOVE[i][ii]], cur_cell[i]

                        else:

                            pass

                            # print

        print(ans)


problem_p00190()

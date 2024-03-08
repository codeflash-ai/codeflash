def problem_p04008():
    from heapq import heappop, heappush

    n, k = list(map(int, input().split()))

    a = [0] + list(map(int, input().split()))

    ans = 0

    if a[1] != 1:

        ans += 1

    links_c = [set() for _ in range(n + 1)]

    for i, ai in enumerate(a[2:], 2):

        links_c[ai].add(i)

    depth = [-1] * (n + 1)

    depth[1] = 0

    stack = [1]

    hq = []

    while stack:

        next = []

        while stack:

            i = stack.pop()

            for j in links_c[i]:

                depth[j] = depth[i] + 1

                heappush(hq, (depth[j] * -1, j))

                next.append(j)

        stack = next[::]

    while hq[0][0] * -1 > k:

        di, i = heappop(hq)

        if depth[i] <= k:

            continue

        for j in range(k - 1):

            i = a[i]

        ans += 1

        links_c[a[i]].remove(i)

        stack = [i]

        while stack:

            next = []

            while stack:

                i = stack.pop()

                depth[i] = 1

                for j in links_c[i]:

                    next.append(j)

            stack = next[::]

    print(ans)

    """
    
    首都の自己ループを除くと、首都を根として、
    
    子→親の有向木になっている。
    
    
    
    ・首都のテレポート先は首都でないとダメ
    
    →　長さ2以上の閉路を作るとうまくいかない
    
    
    
    ・首都以外は、首都までの距離をk以下にできればよい
    
    
    
    もっとも根からの距離が遠い頂点xをとる。
    
    xが根からの距離k以下　→　終了してOK
    
    xが根からの距離k以上
    
    →　xからk-1個離れた親を首都につなぎなおす
    
    　そして再度最も遠い頂点を探す、を繰り返す。
    
    
    
    """


problem_p04008()

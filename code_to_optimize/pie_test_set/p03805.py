def problem_p03805():
    # 頂点および辺の数を入力

    N, M = list(map(int, input().split()))

    # 辺のリストを作成

    list_edge = [[] for _ in range(N + 1)]

    for _ in range(M):

        a, b = list(map(int, input().split()))

        list_edge[a].append(b)

        list_edge[b].append(a)

    # パス（リスト）のリストを初期化

    list_path = [[1]]

    # パスが存在する、かつパスの長さが頂点の数未満である場合に繰り返し

    while list_path != [] and len(list_path[0]) < N:

        # パスのリストから先頭のパスを取り出す

        path = list_path.pop(0)

        # パスの最後尾の頂点から訪問可能な頂点について、、

        for node in list_edge[path[-1]]:

            # その頂点にまだ訪問していない場合、、

            if node not in path:

                # パスのリストに新たなパスを加える

                list_path.append(path + [node])

    # パスの数を出力する

    print((len(list_path)))


problem_p03805()

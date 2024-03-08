def problem_p03458():
    import sys

    import numpy as np

    from collections import Counter

    read = sys.stdin

    N, K = list(map(int, read.readline().split()))

    k = 2 * K

    xyc = np.array(read.read().split(), np.str).reshape(-1, 3)

    x = xyc[:, 0].astype(int)

    y = xyc[:, 1].astype(int)

    c = xyc[:, 2]

    x[c == "B"] -= K

    x %= k

    y %= k

    """
    
    # grid[[1,1],[2,2]]+=1 としてもgrid[1,2]=1であり重複は考慮されない。
    
    # uniqueを使って(x,y)の重複をカウントする
    
    # return_countはver1.9からだった・・・ ダメじゃん
    
    xy, cnt = np.unique(x + y * 10000, return_counts=True)
    
    x = xy % 10000
    
    y = xy // 10000
    
    """

    xy = Counter(list(zip(x.tolist(), y.tolist())))

    cnt = np.array(list(xy.values()), np.int64)

    xy = np.array(list(xy.keys()), np.int64)

    x = xy[:, 0]

    y = xy[:, 1]

    grid = np.zeros((3 * K, 3 * K), np.int64)

    grid[x, y] += cnt

    grid[x + K, y + K] += cnt

    grid[x + K, y] -= cnt

    grid[x, y + K] -= cnt

    np.cumsum(grid, axis=0, out=grid)

    np.cumsum(grid, axis=1, out=grid)

    p1 = grid[:K, :K] + grid[k:, k:] + grid[K:k, K:k] + grid[k:, :K] + grid[:K, k:]

    p2 = grid[K:k, :K] + grid[K:k, k:] + grid[:K, K:k] + grid[k:, K:k]

    print((max(p1.max(), p2.max())))


problem_p03458()

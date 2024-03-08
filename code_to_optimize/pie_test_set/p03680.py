def problem_p03680():
    N = int(eval(input()))

    a = []

    for i in range(N):

        a.append(int(eval(input())))

    # a=[0]*N

    # for i in range(N):

    #   a[i]=int(input())

    count = 0

    # loopの定義→同じインデックスを使用したらloop判定

    # in の処理を爆速で実行するためにsetを使用する

    # loop=[]

    loop = set()

    tmp = 1

    while True:

        tmp = a[tmp - 1]

        count += 1

        if tmp == 2:

            print(count)

            exit()

        if tmp - 1 in loop:

            print((-1))

            exit()

        # loop.append(tmp-1)

        loop.add(tmp - 1)


problem_p03680()

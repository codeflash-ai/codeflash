def problem_p02383():
    currAndRot2FBLR = {
        "N": {"N": "F", "E": "R", "S": "B", "W": "L"},
        "E": {"E": "F", "S": "R", "W": "B", "N": "L"},
        "S": {"S": "F", "W": "R", "N": "B", "E": "L"},
        "W": {"W": "F", "N": "R", "E": "B", "S": "L"},
    }

    dice = {
        "1": {"F": "2F", "R": "4R", "B": "5B", "L": "3L"},
        "2": {"F": "6F", "R": "4F", "B": "1F", "L": "3F"},
        "3": {"F": "6L", "R": "2F", "B": "1R", "L": "5F"},
        "4": {"F": "6R", "R": "5F", "B": "1L", "L": "2F"},
        "5": {"F": "6B", "R": "3F", "B": "1B", "L": "4F"},
        "6": {"F": "5B", "R": "4L", "B": "2F", "L": "3R"},
    }

    faces = list(map(int, input().split()))

    cmd = eval(input())

    ## initial state

    currNum = "1"

    currDir = "N"

    for c in cmd:

        numDir = dice[currNum][currAndRot2FBLR[currDir][c]]

        currNum = numDir[0]

        currFBLR = numDir[1]

        currDir = {v: k for k, v in list(currAndRot2FBLR[currDir].items())}[currFBLR]

    print((faces[int(currNum) - 1]))


problem_p02383()

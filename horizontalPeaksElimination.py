LONG_MEAN_STEP = 35
MEAN_STEP = 15
LITTLE_STEP = 5


def delCoord(coordKL, coordKR, coordAL, coordAR):
    coordKL[0], coordKL[1], coordKL[2] = 0, 0, False
    coordKR[0], coordKR[1], coordKR[2] = 0, 0, False
    coordAL[0], coordAL[1], coordAL[2] = 0, 0, False
    coordAR[0], coordAR[1], coordAR[2] = 0, 0, False


def whoSback(lf, rf, growing, prevBackfoot):
    if growing:
        if prevBackfoot == "rf":
            return "lf" if rf > lf else "rf"
        else:
            return "rf" if lf > rf else "lf"
    else:
        if prevBackfoot == "rf":
            return "rf" if rf > lf else "lf"
        else:
            return "lf" if lf > rf else "rf"


def horizontalPeaks(coordsKL, coordsKR, coordsAL, coordsAR):
    i, deleted, idx, last, jump, countDel, distance = 0, False, 0, 0, False, 1, False
    prevKL, prevKR, prevAL, prevAR, curKL, curKR, curAL, curAR = 0, 0, 0, 0, 0, 0, 0, 0
    for n in range(0, len(coordsKL)):
        if coordsKL[n][2] and coordsKR[n][2]:
            idx = n
            curKL, curKR, curAL, curAR = coordsKL[n][0], coordsKR[n][0], coordsAL[n][0], coordsAR[n][0]
            prevKL, prevKR, prevAL, prevAR = curKL, curKR, curAL, curAR
            break
    # find last relevant item
    for m in reversed(range(len(coordsKL))):
        if coordsKL[m][2]:
            last = m
            break
    direction = coordsKL[idx][0] - coordsKL[last][0]
    # growing if is a left to right walk
    if direction < 0:
        growing = True
    else:
        growing = False
    if growing:
        prevBackFoot = "rf" if curAR < curAL else "lf"
    else:
        prevBackFoot = "rf" if curAR > curAL else "lf"
    i = idx
    while i < len(coordsAL):
        if i == 6:
            print('here')
        # if one coord is missing
        if (not coordsKL[i][2] or not coordsKR[i][2] or not coordsAL[i][2] or not coordsAR[i][2]) and not deleted:
            delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
            deleted = True
            countDel = countDel + 1
        else:
            curKL, curKR, curAL, curAR = coordsKL[i][0], coordsKR[i][0], coordsAL[i][0], coordsAR[i][0]

        if not deleted:
            if curKR < curKL:
                if curAR >= curAL:
                    delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
                    deleted = True
                    countDel = countDel + 1
            else:
                if curAL >= curAR:
                    delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
                    deleted = True
                    countDel = countDel + 1
            if deleted:
                jump = True
            else:
                if not coordsKL[i-1][2]:
                    changed = True if prevBackFoot != whoSback(curAL, curAR, growing, prevBackFoot) else False
                    if changed:
                        jump = True
                        prevBackFoot = "lf" if prevBackFoot == "rf" else "rf"
                        countDel = 1
                        prevKL, prevKR, prevAL, prevAR = curKL, curKR, curAL, curAR
                    else:
                        if not distance:
                            delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
                            deleted, jump = True, True
                            countDel = countDel + 1

        if not deleted and not jump:
            if i == idx:
                prevKL, prevKR, prevAL, prevAR = curKL, curKR, curAL, curAR
            if growing:
                if ((curKL < (prevKL - LITTLE_STEP)) or (curKR < (prevKR - LITTLE_STEP)) or (curAL < (prevAL - LITTLE_STEP)) or (curAR < (prevAR - LITTLE_STEP))) and not deleted:
                    delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
                    deleted, distance = True, True
                    countDel = countDel + 1
                if ((curKL > (prevKL + (countDel*30))) or (curKR > (prevKR + (countDel*30))) or (curAL > (prevAL + (countDel*30))) or (curAR > (prevAR + (countDel*30)))) and not deleted:
                    delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
                    deleted, distance = True, True
                    countDel = countDel + 1
            else:
                if ((curKL > (prevKL + LITTLE_STEP)) or (curKR > (prevKR + LITTLE_STEP)) or (curAL > (prevAL + LITTLE_STEP)) or (curAR > (prevAR + LITTLE_STEP))) and not deleted:
                    delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
                    deleted, distance = True, True
                    countDel = countDel + 1
                if ((curKL < (prevKL - (countDel*30))) or (curKR < (prevKR - (countDel*30))) or (curAL < (prevAL - (countDel*30))) or (curAR < (prevAR - (countDel*30)))) and not deleted:
                    delCoord(coordsKL[i], coordsKR[i], coordsAL[i], coordsAR[i])
                    deleted, distance = True, True
                    countDel = countDel + 1

            if not deleted:
                prevKL, prevKR, prevAL, prevAR = curKL, curKR, curAL, curAR
                countDel, distance = 1, False
            else:
                deleted = False
        else:
            deleted = False
            jump = False
        changed = False
        i = i + 1
    return coordsKL, coordsKR, coordsAL, coordsAR

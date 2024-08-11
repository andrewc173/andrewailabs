import sys; args = sys.argv[1:]
import re
import time
#args = ["dct20k.txt", "5x5", "0"]
#args = ["dct20k.txt", "3x4", "0"]
#args = ["dct20k.txt", "4x3", "0", "V0x2U", "h1x0R"]
#args = ["dct20k.txt", "5x3", "0"]
#args = ["dct20k.txt", "4x4", "0"]
#args = ["dct20k.txt", "5x5", "0", "V0x0Price", "H0x4E"]
#args = ["dct20k.txt", "7x7", "11"]
#args = ["dct20k.txt","4x4", "2", "V0x0R", "V2x0M", "V3x2A"]
#args = ["dctEckel.txt", "9x13", "19", "V0x1DOG"]
#args = ["dct20k.txt", "9x9", "12", "V0x7but", "V6x7cup"]
#args = ["dct20k.txt", "3x3", "0", "V0x0the"]
#args = ["dct20k.txt", "7x7", "11", "H6x0CODY", "H5x6", "v6x6#", "V1x2STARED", "V6x4#"]
#args = ["dct20k.txt", "9x13", "19", "V2x3#", "v5x2", "h5x1", "V5x7", "H4x0"]
args = ["dct20k.txt", "20x20", "70", "v15x2", "h4x2#", "V1x0A", "h1x5E"]
#args = ["dct20k.txt", "7x7", "11", "h2x6", "V6x5", "h0x1", "V1x0UFO", "v0x6#", "H5x1ARREST"]
wordList = open(args[0]).read().split("\n")

global LETTERS
global finalBrd
WORD_FILL = False

LETTERS = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}

# letter_frequencies = {
#     'a': 8.17, 'b': 1.49, 'c': 2.78, 'd': 4.25, 'e': 12.70,
#     'f': 2.23, 'g': 2.02, 'h': 6.09, 'i': 6.97, 'j': 0.15,
#     'k': 0.77, 'l': 4.03, 'm': 2.41, 'n': 6.75, 'o': 7.51,
#     'p': 1.93, 'q': 0.10, 'r': 5.99, 's': 6.33, 't': 9.06,
#     'u': 2.76, 'v': 0.98, 'w': 2.36, 'x': 0.15, 'y': 1.97,
#     'z': 0.07
# }
frequency_dict = {}
total_letters = 0


for word in wordList:
    for letter in word:
        letter = letter.upper()
        if not letter.isalpha():
            continue
        if letter in frequency_dict:
            frequency_dict[letter] += 1
        else:
            frequency_dict[letter] = 1
        total_letters += 1
for letter in frequency_dict:
    frequency_dict[letter] = (frequency_dict[letter] / total_letters) * 100
#print(frequency_dict)

def printBoard(brd, width=None):
    if width == None:
        width = WIDTH
    for start in range(0, len(brd), width):
        print(brd[start:start+width])

def parseArgs(args):
    global HEIGHT
    global WIDTH

    global NUM_BLOCKS
    global brd
    global LENGTH_BOARD
    global WORD_LIST
    global NEW_USED
    global NEW_H_WORDS
    global NEW_V_WORDS

    global NON_WORKING_PATTERN
    global NO_REPLACING_LETTER 

    global PATTERNS

    PATTERNS = {}
    NO_REPLACING_LETTER = 10000000
    WORD_LIST = []
    NEW_USED = []
    NON_WORKING_PATTERN = set()
    NEW_H_WORDS = set()
    NEW_V_WORDS = set()

    #dicFile = arg
    for word in wordList:
        word = word.replace('\n', '').strip().upper() 
        if re.match('^[A-Za-z]+$', word):
            WORD_LIST.append(word.upper())
    for arg in args:
        #print("arg:", arg)
        idxToChange = []
        numToChange = 0
        y = -1 
        x = -1
        if re.match('\w+\.txt$', arg):
            pass
            # dicFile = arg
            # with open(arg, 'r') as file: 
            #         word_list = file.readlines()
            # #print(word_list)
            # for word in word_list:
            #     word = word.replace('\n', '').strip().upper() 
            #     #print(word)
            #     if re.match('^[A-Za-z]+$', word):
            #         #print("adding", word)
            #         WORD_LIST.append(word.upper())
        elif re.match('^\d+x\d+$', arg):
            #print("REACHES HERE")
            xbyx = arg
            hw = xbyx.split("x")
            HEIGHT=int(hw[0])
            WIDTH=int(hw[1])
            brd = "-"*HEIGHT*WIDTH
            LENGTH_BOARD = HEIGHT*WIDTH
            #print("brd", brd)
        elif re.match('^\d*$', arg):
            NUM_BLOCKS = int(arg)
        elif arg[0].upper() == "V":
            dimenStr = arg[1:]
            if "x" in arg:
                dimensStrArr = dimenStr.split("x", 1)
            elif "X" in arg:
                dimensStrArr = dimenStr.split("X", 1)
            y = int(dimensStrArr[0])
            startOfWordLetters = 0
            if dimensStrArr[1][0].isdigit():
                startOfWordLetters = 1
            if len(dimensStrArr[1]) >= 2 and dimensStrArr[1][1].isdigit():
                startOfWordLetters = 2
            x = int(dimensStrArr[1][:startOfWordLetters])
            idx = y*WIDTH+x
            numToChange = len(dimensStrArr[1][startOfWordLetters:])
            word = dimensStrArr[1][startOfWordLetters:]
            for a in range(idx, idx + (numToChange) * WIDTH, WIDTH):
                idxToChange.append(a)
        elif arg[0].upper() == "H":
            dimenStr = arg[1:]
            if "x" in dimenStr:
                dimensStrArr = dimenStr.split("x")
            elif "X" in dimenStr:
                dimensStrArr = dimenStr.split("X")
            y = int(dimensStrArr[0])
            startOfWordLetters = 0
            if dimensStrArr[1][0].isdigit():
                startOfWordLetters = 1
            if len(dimensStrArr[1]) >= 2 and dimensStrArr[1][1].isdigit():
                startOfWordLetters = 2
            x = int(dimensStrArr[1][:startOfWordLetters])
            idx = y*WIDTH+x
            
            numToChange = len(dimensStrArr[1][startOfWordLetters:])
            word = dimensStrArr[1][startOfWordLetters:]
            for a in range(idx, idx + (numToChange)):
                idxToChange.append(a)
        for i in range(0, numToChange):
            brd = brd[:idxToChange[i]] + word[i].upper() + brd[idxToChange[i]+1:]
        if numToChange == 0 and (arg[0].upper() == "V" or arg[0].upper() == "H"):
            brd = brd[:idx] + "#" + brd[idx+1:]
    #print("brd:", brd)
    #print("dicFile:", dicFile)
    #print("board:", brd)
    #print("wordList:", WORD_LIST)

def makeBlocksSymmetric(brd):
    board_list = list(brd)
    for idx, char in enumerate(board_list):
        if char == "#":
            sym_idx = LENGTH_BOARD - 1 - idx
            if board_list[sym_idx] == "-":
                board_list[sym_idx] = "#"
    return ''.join(board_list)

def isSymmetric(brd):
    #print("testing symmetry on:")
    #printBoard(brd)
    for idx in range(0, (LENGTH_BOARD)//2):
        symA = brd[idx]
        symB = brd[LENGTH_BOARD-1-idx]
        if symA not in LETTERS and symB not in LETTERS and symA != symB:
            return False
    #print("it is symmetric")
    return True

def floodFill(brd, start):
    #print("calling floodFill")
    def to2D(index):
        return index // WIDTH, index % WIDTH
    
    def to1D(row, col):
        return row * WIDTH + col

    visited = set()
    #printBoard(brd)
    def dfs(row, col):
        if row < 0 or row >= HEIGHT or col < 0 or col >= WIDTH or (brd[to1D(row, col)] == "#") or to1D(row, col) in visited:
            return
        visited.add(to1D(row, col))
        dfs(row - 1, col)  # up
        dfs(row + 1, col)  # down
        dfs(row, col - 1)  # left
        dfs(row, col + 1)  # right

    start_row, start_col = to2D(start)
    dfs(start_row, start_col)
    
    return visited

    
def isConnected(brd):
    #find the first dash
    for i in range(LENGTH_BOARD):
        if brd[i] == "-":
            #print("first dash is at", i)
            indexes_of_dash = {index for index, character in enumerate(brd) if character == "-" or character in LETTERS}
            floodDashesSet = floodFill(brd, i)
            return indexes_of_dash == floodDashesSet
    return True

def make_segments_set(brd):
    segments_set = set()
    for row in range(HEIGHT):
        row_str = brd[row * WIDTH : (row + 1) * WIDTH]
        segments_set.update(row_str.split("#"))
    for col in range(WIDTH):
        col_str = ''.join([brd[row * WIDTH + col] for row in range(HEIGHT)])
        segments_set.update(col_str.split("#"))
    
    if '' in segments_set:
        segments_set.remove('')
    
    return segments_set

def allThreeLetterWords(brd):
    segments_set = make_segments_set(brd)
    for segment in segments_set:
        if len(segment) < 3:
            return False
    return True

def extract_words(brd, HEIGHT, WIDTH):
    horizontal_words = []
    vertical_words = []

    for row in range(HEIGHT):
        row_str = brd[row * WIDTH:(row + 1) * WIDTH]
        horizontal_words.extend([word for word in row_str.split("#") if word])

    for col in range(WIDTH):
        col_str = ''.join(brd[row * WIDTH + col] for row in range(HEIGHT))
        vertical_words.extend([word for word in col_str.split("#") if word])

    return horizontal_words, vertical_words

def checkWordConditions(brd):
    def in_bounds(row, col):
        return 0 <= row < HEIGHT and 0 <= col < WIDTH
    
    def to_2D(index):
        return divmod(index, WIDTH)
    
    def is_valid_char(char):
        return char.isalpha() or char in ['~', '-']

    def check_3_letter(row, col, delta_row, delta_col):
        count = 1
        step = 1
        while in_bounds(row + step * delta_row, col + step * delta_col) and is_valid_char(brd[(row + step * delta_row) * WIDTH + (col + step * delta_col)]):
            count += 1
            if count == 3:
                return True
            step += 1
        step = -1
        while in_bounds(row + step * delta_row, col + step * delta_col) and is_valid_char(brd[(row + step * delta_row) * WIDTH + (col + step * delta_col)]):
            count += 1
            if count == 3:
                return True
            step -= 1
        
        return False
    
    for index, char in enumerate(brd):
        if is_valid_char(char): 
            row, col = to_2D(index)
            if not (check_3_letter(row, col, 0, 1) and check_3_letter(row, col, 1, 0)):
                return False
    return True
def eachLetterInWord(brd):
    segments_set = make_segments_set(brd)

def isSolved(brd):
    return brd.count("#") == NUM_BLOCKS

def fillLine(line):
    new_line = []
    i = 0
    while i < len(line):
        if line[i] == "-":
            gap_start = i
            while i < len(line) and line[i] == "-":
                i += 1
            gap_end = i
            #print("gap_start idx =", gap_start)
            #print("gap_end idx =", gap_end)
            gap_length = gap_end - gap_start
            valid_gap = (gap_start == 0 or (gap_start >= 1 and line[gap_start-1] == "#")) and (gap_end >= len(line) or line[gap_end] == "#")
            if valid_gap and 0 < gap_length <= 2:
                new_line.extend(["#"] * gap_length)
            else:
                new_line.extend(["-"] * gap_length)
        else:
            new_line.append(line[i])
            i += 1
    return new_line

def isInvalid(brd):
    #print("Symmetric:", isSymmetric(brd))
    #print("Connected:", isConnected(brd))
    #print("3-letter-words:", allThreeLetterWords(brd))
    #print("letters-in-2-words:", checkWordConditions(brd))
    return not(isSymmetric(brd) and isConnected(brd) and allThreeLetterWords(brd) and checkWordConditions(brd))

def neverValid(brd):
    segments_set = make_segments_set(brd)
    for segment in segments_set:
        if len(segment) < 3 and any(c in LETTERS for c in segment):
            #print("returns neverValid cuz of segment", segment)
            return True
    #print(isConnected(brd))
    #return False
    return not isConnected(brd)

def fillGaps(brd):
    origBrd = brd

    board_2d = [list(brd[i * WIDTH:(i + 1) * WIDTH]) for i in range(HEIGHT)]

    board_2d = [fillLine(row) for row in board_2d]
    for col in range(WIDTH):
        column = [board_2d[row][col] for row in range(HEIGHT)]

        filled_column = fillLine(column)
        for row in range(HEIGHT):
            board_2d[row][col] = filled_column[row]
    toRet = ''.join([''.join(row) for row in board_2d])
    toRet = makeBlocksSymmetric(toRet)
    if origBrd == toRet:
        return toRet
    else:
        return fillGaps(toRet)
def tranposedIdx(index, width):
    row = index // width
    col = index % width
    transposed_index = col * width + row
    return transposed_index


def border_the_board(board):
    return "#"*(WIDTH+2) + "".join(["#" + board[start:start+WIDTH] + "#" for start in range(0, len(board), WIDTH)]) + "#"*(WIDTH+2)
def calculate_min_segment_length(board, index):
    score = 0
    left_length = 0
    right_length = 0
    bordered_board = border_the_board(board)
    index = (WIDTH+2)*(index//WIDTH + 1) + (index%WIDTH+1) #newindex
    i = index - 1
    while board[i] != '#':
        left_length += 1
        i -= 1

    # Calculate right segment length
    i = index + 1
    while board[i] != '#':
        right_length += 1
        i += 1
    score += min(left_length, right_length)
    newTempBoard = transpose(board, WIDTH+2)
    newIdx = tranposedIdx(i, WIDTH+2)
    left_length = 0
    right_length = 0
    i = newIdx - 1
    while i >= 0 and newTempBoard[i] != '#':
        left_length += 1
        i -= 1
    i = newIdx + 1
    while i < len(board) and newTempBoard[i] != '#':
        right_length += 1
        i += 1
    score += min(left_length, right_length)
    return score

HUER = {}
def huerestic(brd):
    return HUER[brd]
def setOfChoices(brd):
    choices = set()
    #choices = []
    # Iterate only up to half the length to ensure symmetry
    for i in range(LENGTH_BOARD // 2 + 1):
        symmetric_idx = LENGTH_BOARD - 1 - i
        #print("INDEX, SYMMETRIC INDEX:", i, symmetric_idx)
        if i == symmetric_idx and brd[i] == "-":
            new_brd = brd[:i] + "#" + brd[symmetric_idx+1:]
            #HUER[new_brd] = calculate_min_segment_length(brd, i)
            choices.add(new_brd)
            #print("ADDS THE SYMETRY BOARD:", new_brd)
        elif brd[i] == "-" and brd[symmetric_idx] == "-":
            # Replace both symmetric "-" with "#"
            new_brd = brd[:i] + "#" + brd[i+1:symmetric_idx] + "#" + brd[symmetric_idx+1:]
            #HUER[new_brd] = calculate_min_segment_length(brd, i)
            choices.add(new_brd)
            #choices += new_brd
    #return choices
    listChoices = list(choices)
    #listChoices.sort(key=lambda x: heuristic(x), reverse=True)  # Sort by heuristic value, high to low
    return listChoices

def bruteForce(brd):
    #do a visited
    #print("TRYING BRUTE FORCE ON:") ########## comment out
    #printBoard(brd) ############################ comment out
    brd = fillGaps(brd)
    #print("first, fills gaps, so board now becomes:")
    #printBoard(brd)
    if brd.count("#") > NUM_BLOCKS:
        return ""
    if neverValid(brd):
        #print("However, this board is NEVER VALID, so returns [empty string]") ########## comment out
        return ""
    if isSolved(brd): 
        #print("the right number of #'s....") ######### comment out
        if isInvalid(brd):
            #print("This board, even with the right number of #'s, is invalid.") ########### comment out
            return ""
        else:
            #print("it is valid, so returns board") ########## comment out
            return brd
    choicesSet = setOfChoices(brd)
    #print("BOARD IS NOT SOLVED, so tries bruteForce on following sets") ########## comment out
    #print("choices set:", choicesSet) ########### comment out
    for possibleChoice in choicesSet:
        subBrd = possibleChoice
        #print("subBrd=", subBrd) ########## comment out
        #printBoard(subBrd)
        subBrd = fillGaps(subBrd)
        if subBrd.count("#") > NUM_BLOCKS: 
            continue
        bF = bruteForce(subBrd)
        if bF: return bF
    return ""

def update_row_with_word(row, word, start_pos):
    return row[:start_pos] + word + row[start_pos+len(word):]


#PUT THIS PART AT THE BEGINNING OF CODE
def generate_dictionary(n): #Generates the patterns dcitionary for word length n
    dictionary = {}
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    dictionary['_'*n] = set()
    for i in range(n):
        prefix = '_' * i
        suffix = '_' * (n - 1 - i)
        for letter in alphabet:
            key = prefix + letter + suffix
            dictionary[key] = set()
    for word in WORD_LIST:
        if len(word) == n:
            dictionary['_'*n].add(word)
            for j in range(n):
                prefix = '_' * j
                suffix = '_' * (n-1-j)
                key = prefix + word[j] + suffix
                if key in dictionary:
                    dictionary[key].add(word)

    return dictionary

def createWordPattenList(brd, width, direction):
    height = len(brd) // width
    pattern_list = {} 
    for row_index in range(height):
    #print("WIDTH", WIDTH)
    #print("HEIGHT", HEIGHT)
        row_str = brd[row_index * width : (row_index + 1) * width]
#        print("old row_str:", row_str) ###################
        start = 0 
        key = start
        while start < width:
            if row_str[start] == '#': 
                start += 1
                continue 

            end = start
            while end < width and row_str[end] != '#': 
                end += 1
            slot = row_str[start:end]
            
            #print("Patten List", slot_pattern)
            #pattern_list.append(slot_pattern) 
            pattern_list[direction + str(start + row_index * width)] = slot
            start = end  
 #   print(pattern_list)
    return pattern_list

def transpose(board, width):
    return "".join(["".join([board[idx] for idx in range(0, len(board)) if idx%width==i]) for i in range(0, width)])

def createVwordPattenList(board):
    transposeBoard = transpose(board, WIDTH) 
    # print(transposeBoard)
    # printBoard(transposeBoard, HEIGHT)
    pattern_list = createWordPattenList(transposeBoard, HEIGHT, "V")
    return pattern_list

# def findPossibleWords(slot, wordlist):
#     slot_pattern = slot.replace('-', '.')
#     possibleWords = []

#     pattern = re.compile(f'^{slot_pattern}$')

#     for word in wordlist:        
#         match = pattern.search(word)
#         if match:
#             possibleWords.append(word.upper())
#     return possibleWords

def h2(word):
    score = 0
    for letter in word:
        score += frequency_dict[letter.upper()]
    return score

def findPossibleWords(slot, wordlist):
    possiblePatterns = []
    #splits the slot into its individual patterns
    for i in range(len(slot)): 
        if slot[i].isalpha():
            possiblePatterns.append("_"*i + slot[i] + "_"*(len(slot)-1-i))
    if len(possiblePatterns) == 0:
        possiblePatterns.append("_"*len(slot))
    #for each pattern, intersects the set to combine individual patterns together
    possibleWords = PATTERNS[possiblePatterns[0]]
    #print(possibleWords)
    for pattern in possiblePatterns:
        possibleWords = possibleWords.intersection(PATTERNS[pattern])
    
    my_list = list(possibleWords)
    #my_list.sort(key=lambda word: heuristic(word), reverse=True)

    return my_list 


def updateBoard(brd, location, word, direction):
    
    #print("update board with the word: " + word)
    index = int(location[1:])
    

    if direction == 'H':
        brd = brd[:index] + word + brd[(index + len(word)):]
    else:
        #printBoard(brd)
        brd = transpose(brd, WIDTH)
        #printBoard(brd, HEIGHT)
        brd = brd[:index] + word + brd[(index + len(word)):]
        #printBoard(brd, HEIGHT)
        brd = transpose(brd, HEIGHT)
        #printBoard(brd, WIDTH)
    return brd

def isAnyInvalidWords(board):
    pattern_dict_H = createWordPattenList(board, WIDTH, 'H')
    pattern_dict_V = createVwordPattenList(board)
    combined_dict = {**pattern_dict_H, **pattern_dict_V}
    for value in (combined_dict).values():  
        if value.isalpha():
            if value in NEW_USED or value in WORD_LIST:
                continue
            else:
                return True
    return False   

def initScan(board):
    pattern_dict_H = createWordPattenList(board, WIDTH, 'H')
    pattern_dict_V = createVwordPattenList(board)
    for value in (pattern_dict_H).values():  
        if value.isalpha():            
            if value in WORD_LIST:
                NEW_USED.append(value)
        else:
            NEW_H_WORDS.update(findPossibleWords(value, WORD_LIST))
    
    for value in (pattern_dict_V).values():  
        if value.isalpha():            
            if value in WORD_LIST:
                NEW_USED.append(value)
        else:
            NEW_V_WORDS.update(findPossibleWords(value, WORD_LIST))  

def countReplacingLetters(word):
    #replacingChar = '.'
    replacingChar = '-'
    if word.count(replacingChar) == 0:        
        return NO_REPLACING_LETTER
    if word.count(replacingChar) == len(word):
        return NO_REPLACING_LETTER - 1 #  Do not select full dots 
    return word.count(replacingChar)




def countLen(word):
    replacingChar = '-'
    if word.count(replacingChar) == 0:        
        return -1
    if word.count(replacingChar) == len(word):
        return 0 
    return len(word)  

def getMostConstrained(pattern_dict):
    #print("pattern_dict:", pattern_dict)
    #sorted_items = None
    min_length = None
    most_constrained_key = None

    for key in pattern_dict:
        if pattern_dict[key].count("-") == 0:
            continue
        possible_words = findPossibleWords(pattern_dict[key], WORD_LIST)
        current_length = len(possible_words)
        if min_length is None or current_length < min_length:
            min_length = current_length
            most_constrained_key = key

    return most_constrained_key
    # if max(HEIGHT, WIDTH) <= 5:
    #     sorted_items = sorted(pattern_dict.items(), key=lambda item: countReplacingLetters(item[1]), reverse=False)
    # else:
    #     sorted_items = sorted(pattern_dict.items(), key=lambda item: (countLen(item[1]), -(item[1].count('-'))), reverse=True)
            
    # Extract and print the sorted patterns for visualization
    
    #print(sorted_items)
    
    # return sorted_items[0][0]

def fillWords(cBrd):
    printBoard(cBrd)
    print("\n")
    # oldBrd = cBrd
    global WORD_FILL 
    global finalBrd 
    # if newBrd != None:
    #     brd = newBrd
    if cBrd.count("-") == 0:
        WORD_FILL = True
        finalBrd = cBrd
        #print(finalBrd)
        return finalBrd

        return cBrd
    pattern_dict_H = createWordPattenList(cBrd, WIDTH, 'H')
    pattern_dict_V = createVwordPattenList(cBrd)
    pattern_dict_all = {**pattern_dict_H, **pattern_dict_V}
    most_constrained_key = getMostConstrained(pattern_dict_all)
    #vContrainedIndex= getMostConstrained(pattern_dict_V)

    isHorizon = True
    if most_constrained_key[0] == "V":
        isHorizon = False
    possibleWords = []
    wordStart = most_constrained_key
    #hPattern = pattern_dict_H[hContrainedIndex]
    #vPattern = pattern_dict_V[vContrainedIndex]

    #wordStart = hContrainedIndex
    wordPatten = pattern_dict_all[most_constrained_key]



    # hDotCount = countReplacingLetters(hPattern)    
    # vDotCount = countReplacingLetters(vPattern)

    #if (hDotCount == NO_REPLACING_LETTER and vDotCount == NO_REPLACING_LETTER):

    #longest word 
    # if (len(hPattern) == len(vPattern)):
    #     if hPattern.count('-') > vPattern.count('-'):
    #         wordPatten = vPattern
    #         wordStart = vContrainedIndex
    #         isHorizon = False

    # if len(hPattern) < len(vPattern):
    #     wordPatten = vPattern
    #     wordStart = vContrainedIndex
    #     isHorizon = False

    #print("--- New words pattern: " + wordPatten)

    if (wordPatten in NON_WORKING_PATTERN):
        #print("NON_WORKING_PATTERN:" + str(NON_WORKING_PATTERN))
        if NEW_USED:
            
            NEW_USED.pop()
        return cBrd

    possibleWords = [] 
    if isHorizon:
        possibleWords = findPossibleWords(wordPatten, NEW_H_WORDS)
    else:
        possibleWords = findPossibleWords(wordPatten, NEW_V_WORDS)
    possibleWords.sort(key=lambda word: h2(word), reverse=True)

    if len(possibleWords) == 0:
        NON_WORKING_PATTERN.add(wordPatten)  
        if NEW_USED:
            NEW_USED.pop()
        return cBrd
    
#    print(possibleWords)
    
    new_used_len_stamp = len(NEW_USED)
    
    for word in possibleWords:        
        if (WORD_FILL):
            break
        
        if word.upper() in NEW_USED:
            continue
        
        if isHorizon:
            cBrd = updateBoard(cBrd, wordStart, word, "H")
        else:
            cBrd = updateBoard(cBrd, wordStart, word, "V")

        # print("~~~~~~ After Update ~~~~~~~~")
        printBoard(cBrd)
        if isAnyInvalidWords(cBrd):  # check new bloard if there  invalid word
            if NEW_USED:   # this one has a bug
                NEW_USED.pop()
            return cBrd
        
        NEW_USED.append(word)
#        print(NEW_USED)
        
        fillWords(cBrd)

    if len(NEW_USED) == new_used_len_stamp:
        if NEW_USED:
            
            NEW_USED.pop() #all childred does not add new word, make progress
    return cBrd
    
parseArgs(args)

start_time = time.time()  # Start time

brd = makeBlocksSymmetric(brd)

brd = bruteForce(brd)

for i in range(3, max(HEIGHT, WIDTH)+1):
    dic = generate_dictionary(i) #generates patterns dict for length height
    #PATTERNS = PATTERNS | dic 
    PATTERNS = {**PATTERNS, **dic}

if(brd != ""):
    finalBrd = ""

    # print(brd)
    #printBoard(brd)
    #print("\n")
    initScan(brd)
    brd = fillWords(brd)
    if (finalBrd == "" or None):
        pass
    printBoard(finalBrd)

end_time = time.time()  # End time
print(f"Program running time: {end_time - start_time} seconds")

#andrew chen pd 6 2026
import sys; args = sys.argv[1:]
import re
import math
#args = "99 B26 B69 B64 B91 B78 B62 B52 B44 B46 B28 B47N B53S B26E B55S B60S B81W B60W B36N B91N B50S B58E B1S B71N B81E B50W R6:62 R68:9 R52:60 R64:39 R29:50 R71:77 R1:87 R72:74 R63:79 R2:31 R4:82 R46:28 R44:66 R47:38 R95:10 R79:11 R85:93 R74:53 R69:49 R10:76".split(" ")
#args = "99 R49:39 R2:21 R96:47 R30:14 R72:62 R52:38 R12:61 R90:34 R82:11 R55:80 R85:20 R20:92 R62:82 R1:94 R45:79 R92:70 R60:32 R87:36 R84:14 R91:59 R80:1 R7:92 R23:62 R40:54 R4:89 R64:72 R19:74 R81:43 R8:25 R67:81 R63:6 R14:4 R22:82 R65:40 R98:9 R56:64 R31:63 R76:29 R39:65 R68:68".split(" ")
args = "45 R:5 R10 R14:500 B42 B23 B3 B15 B40 B21 B2 B41 G0 R9 R40 B4E B13S B42E B34S B12S".split(" ")
def isGraphDirective(arg):
    return arg[0] == "G"
def isNumber(arg):
    return re.search(r'^\d+$', arg, re.IGNORECASE)
def isSetImpliedReward(arg):
    return len(arg) >= 2 and arg[0:2] == "R:"
def isSetRewardToDefault(arg):
    return re.match(r'^R\d+$', arg, re.IGNORECASE)
def isSetSpecificRewardForCell(arg):
    return re.match(r'^R(\d+):(\d+)$', arg, re.IGNORECASE)
def isBlockSingle(arg):
    return re.match(r'^B(\d+)$', arg, re.IGNORECASE)
def isBlockDirections(arg):
    return re.match(r'^B(\d+)([NSEW]+)$', arg, re.IGNORECASE)

def defaultWidth(size): #
    sqrtLen = math.sqrt(size)
    length = (int)(sqrtLen)
    while ((size % length != 0)):
        length -= 1
    width = size//length
    return width

def getImpliedRewardFromArg(arg):
    return int(arg[2:])
def getCellToSetImpliedReward(arg):
    return int(arg[1:])
def getCellReward(arg):
    match = re.match(r'^R(\d+):(\d+)$', arg, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return -1, -1
def getBlockSingle(arg):
    return int(arg[1:])
def getBlockDirections(arg):
    match = re.match(r'^B(\d+)([NSEW]+)$', arg, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2)
    else:
        return -1, "-1"


def createGraph(size, width, length):
    adj_list = [[] for _ in range(size)]
    for i in range(size):
        row = i // width
        col = i % width
       #print("i", i, "row", row, "col", col)
        if col > 0:  #connect left 
            adj_list[i].append(i - 1)
        if col < width - 1:  #connect right
            adj_list[i].append(i + 1)
        if row > 0:  #connect up
            adj_list[i].append(i - width)
        if row < length - 1: #connect down
            adj_list[i].append(i + width)
    vprops = {i: -1 for i in range(size)}
    return {
        'size': size,
        'default_rwd': 12,
        'width': width,
        'length': length,
        'zero': False,
        'vprops': vprops,
        'adj_list': adj_list
    }

def setDefaultReward(graph, reward):
    graph['default_rwd'] = reward
    return graph

def setWidth(graph, width):
    graph['width'] = width
    graph['length'] = graph['size']//width
    return graph

def setRewardOfVertex(graph, vertex, reward):
    graph['vprops'][vertex] = reward
    return graph
def removeEdge(graph, v1, v2):
    graph['adj_list'][v1] = [end for end in graph['adj_list'][v1] if end != v2]
    graph['adj_list'][v2] = [end for end in graph['adj_list'][v2] if end != v1]
    return graph

def addEdge(graph, v1, v2):
    graph['adj_list'][v1].append(v2)
    graph['adj_list'][v2].append(v1)
    return graph
     
def originallyConnected(graph, v1, v2):
    width = graph['width']
    return abs(v1 - v2) == width or (abs(v1 - v2) == 1 and v1//width == v2//width) 

def edgeExistsInGraph(graph, v1, v2):
    for vtx in graph['adj_list'][v1]:
        if vtx == v2:
            return True
    return False
def complementEdge(graph, v1, v2):
    edge_exists = edgeExistsInGraph(graph, v1, v2)
    if edge_exists:
        graph = removeEdge(graph, v1, v2)
    else: #edge doesn't currently existed
        if originallyConnected(graph, v1, v2):
            graph = addEdge(graph, v1, v2) 
    return graph

def toggleEdge(graph, v1, v2):
    edge_exists = edgeExistsInGraph(graph, v1, v2)
    if not edge_exists:
        graph = addEdge(graph, v1, v2)
    else:
        graph = removeEdge(graph, v1, v2)
    return graph

def followDirections(startVertex, cardinalDirections, width, length):
    row = startVertex // width
    startRow = row
    col = startVertex % width
    startCol = col
    newVertexes = []
    for direction in cardinalDirections:
        row = startRow
        col = startCol
        if direction == 'N': 
            if row == 0:  
                continue
            row -= 1
        elif direction == 'S':
            if row == length - 1:  
                continue
            row += 1
        elif direction == 'E':  
            if col == width - 1: 
                continue
            col += 1
        elif direction == 'W':  
            if col == 0:  
                continue
            col -= 1
        newVertex = row * width + col
        newVertexes.append(newVertex)
    return newVertexes

def grfParse(lstArgs): 
    graph = None
    zero = True
    impliedReward = 12
    for i, arg in enumerate(lstArgs):
        if isNumber(arg) and i == 0: #size
            size = int(arg)
            width = defaultWidth(size)
            length = size//width
            graph = createGraph(size, width, length) #this is repetitive if we have a width directive
            #print("GRAPH:", graph)
        elif isNumber(arg) and i == 1: #width
            newWidth = int(arg)
            newLength = size//newWidth
            graph = createGraph(size, newWidth, newLength)
        elif isGraphDirective(arg): #G0 or G1
            print(arg)
            if arg[1] == "0":
                graph['zero'] = True
            elif arg[1] == "1":
                graph['zero'] = False
        elif isSetImpliedReward(arg): #form R:#
            impliedReward = getImpliedRewardFromArg(arg)
            graph = setDefaultReward(graph, impliedReward)
        elif isSetRewardToDefault(arg): #form R#
            cell = getCellToSetImpliedReward(arg)
            impliedReward = graph['default_rwd'] 
            graph = setRewardOfVertex(graph, cell, impliedReward)
        elif isSetSpecificRewardForCell(arg): #form R#:#
            cell, reward = getCellReward(arg)
            graph = setRewardOfVertex(graph, cell, reward)
        elif isBlockSingle(arg): #form B#
            v1 = getBlockSingle(arg)
            non_vertices = [node for node in range(size) if node != v1]
            for v2 in non_vertices:
                graph = complementEdge(graph, v1, v2)
        elif isBlockDirections(arg): #form B#[NSEW]#
            startVertex, directions = getBlockDirections(arg)
            width = graph['width']
            length = graph['length']
            endVertexes = followDirections(startVertex, directions, width, length)
            modifyEdgeList = []
            for endVertex in endVertexes:
                if endVertex == -1: continue
                modifyEdgeList.append((startVertex, endVertex))
            for edge in modifyEdgeList:
                startVertex = edge[0]
                endVertex = edge[1]
                graph = toggleEdge(graph, startVertex, endVertex)
    #print("graph:", graph)
    return graph

def print_graph(graph):
    adjacency_list = graph['adj_list']
    for i, edges in enumerate(adjacency_list):
        if edges:
            connections = ', '.join(f"{endVertex} (weight {weight})" for endVertex, weight in edges)
            print(f"Vertex {i} -> {connections}")
        else:
            print(f"Vertex {i} has no connections")

def formatPolicyStr(directions, width):
    result = ""
    for index in range(len(directions)):
        result += directions[index]
        if (index + 1) % width == 0 and index+1 != len(directions):
            result += "\n"
    return result

def findMaxRewardReachable(graph, start):
    isZero = graph['zero']
    maxReward = -1
    distForThat = -1
    if graph['vprops'][start] != -1:
        return graph['vprops'][start], 0
    visited = [False] * graph['size']
    queue = [(start, 0)]  #(current node, current distance)
    visited[start] = True
    qIndex = 0 #pointer to current index in queue

    while qIndex < len(queue):
        #print("CURRENT NODE:", queue[qIndex][0])
        #print("QUEUE:", queue)
        current, dist = queue[qIndex]
        qIndex += 1
        
        if graph['vprops'][current] != -1: #
            #(if it's G1) dividing by (dist + 1) is EXTREMELY IMPORTANT!!! because we are finding the distance from one step away from the intended node,rather than the node
            currRwd = graph['vprops'][current] if isZero else graph['vprops'][current]/(dist+1)  
            if currRwd > maxReward:
                maxReward = currRwd
                distForThat = dist
        else:
            #print(graph['adj_list'][current])
            for neighbor in graph['adj_list'][current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
    return maxReward, distForThat
def determineInitialDirections(graph, width, size):
    height = size//width
    directions = {}
    isZero = graph['zero']
    for i in range(graph['size']):
        if graph['vprops'][i] != -1:
            directions[i] = '*'
            continue
        max_reward = -1
        min_dist = 1000
        initial_moves = set()
        #check 4 cardinal directions
        for dx, dy, dir_code in [(0, -1, 'U'), (0, 1, 'D'), (1, 0, 'R'), (-1, 0, 'L')]:
            x, y = i % width, i // width
            nx, ny = x + dx, y + dy
            neighbor = ny*width + nx
            dx = neighbor - x
            dy = neighbor - y
            if not edgeExistsInGraph(graph, i, neighbor):
                continue
            if 0 <= nx < width and 0 <= ny < height:
                if neighbor < graph['size']:
                    reward, distForThat = findMaxRewardReachable(graph, neighbor)
                    if reward != -1:
                        if reward > max_reward:
                            max_reward = reward
                            min_dist = distForThat
                            initial_moves = {dir_code}
                        elif reward == max_reward:
                            if isZero:
                                if distForThat < min_dist:
                                    initial_moves = {dir_code}
                                    min_dist = distForThat
                                elif distForThat == min_dist:
                                    initial_moves.add(dir_code)
                            else:
                                initial_moves.add(dir_code)
        #print("initial moves:", initial_moves)
        if max_reward== -1 or len(initial_moves) == 0:
            directions[i] = '.'
        else:
            dir_str = ''.join(sorted(initial_moves))
            dir_map = {'RU': 'V', 'DRU': 'W', 'DR': 'S', 'DLR': 'T', 'DL': 'E', 'DLU': 'F',
                       'LU': 'M', 'LRU': 'N', 'DU': '|', 'LR': '-', 'DLRU': '+'}
            directions[i] = dir_map.get(dir_str, dir_str) if len(dir_str) > 1 else dir_str
    return directions

def printVProps(vprops):
    print("Vertex Rewards:", ", ".join([f"{vertex} (rwd {value})" for vertex, value in vprops.items() if value != -1]))
def printGridGraph(adj_list, width):
    height = (len(adj_list) + width - 1) // width 

    for i in range(height):
        current_row = ""
        bottom_row = ""
        for j in range(width):
            node_index = i * width + j
            if node_index >= len(adj_list):  
                break
            node_str = f"[{node_index:02d}]"
            if j < width - 1 and node_index + 1 < len(adj_list) and (node_index + 1) in adj_list[node_index]:
                node_str += " -- "
            else:
                node_str += "    "
            current_row += node_str

            if i < height - 1 and node_index + width < len(adj_list) and (node_index + width) in adj_list[node_index]:
                bottom_row += "  |     "
            else:
                bottom_row += "        "
        print(current_row.rstrip())
        if i < height - 1:
            print(bottom_row.rstrip())
        
def printGridGraphReward(adj_list, vprops, width):
    height = (len(adj_list) + width - 1) // width 
    for i in range(height):
        current_row = ""
        bottom_row = ""
        for j in range(width):
            node_index = i * width + j
            if node_index >= len(adj_list):  
                break
            reward = vprops[node_index]
            if reward == -1: reward = 0
            node_str = f"[{reward:02d}]"
            if j < width - 1 and node_index + 1 < len(adj_list) and (node_index + 1) in adj_list[node_index]:
                node_str += " -- "
            else:
                node_str += "    "
            current_row += node_str

            if i < height - 1 and node_index + width < len(adj_list) and (node_index + width) in adj_list[node_index]:
                bottom_row += "  |     "
            else:
                bottom_row += "        "
        print(current_row.rstrip())
        if i < height - 1:
            print(bottom_row.rstrip())
def main():
    graph = grfParse(args)
    #print(findMaxRewardReachable(graph, 3))
    printVProps(graph['vprops'])
    directions = determineInitialDirections(graph, graph['width'], graph['size'])
    print(formatPolicyStr(directions, graph['width']))
    print("======")
    #printGridGraph(graph['adj_list'], graph['width'])

    printGridGraphReward(graph['adj_list'], graph['vprops'], graph['width'])
    #print(findMaxRewardReachable(graph, 13))

    #print(findMaxRewardReachable(graph, 18))
    #print(findMaxRewardReachable(graph, 19))



if __name__ == '__main__': main()


#andrew chen pd6 2026
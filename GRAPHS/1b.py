import sys; args = sys.argv[1:]
import re
import math
global WIDTH
global LENGTH
global SIZE
N_TYPE = False
#wordList = open(args[0]).read().split("\n")
#args = ["G14", "V-5:-1:-9B"]
#args = ["GG14W14", "V6"]
#args = ["G28W7", "V5:17:3,26,1,21,2:22:4B"]
#args = ["GG3", "E~0=1"]
#args = ["GG12", "E~10,10=-3,-6"]
#args = ["G12", "E~9,9~-2,10"]
#args = ["GG6"]
#args = ["GG11", "E~6=4"]
#args = ["GG15W15", "E~11=9"]
#args = ["GG7", "E~4=2"]
#args = ["G15W3", "E7N="]
#args = ["GG54", "E21S="]
#args = ["GG20", "E~8NW="]
#args = ["GG54", "V-7R5"]
#args = ["GN18"]
#args = ["GG35", "V9R8B", "E~26,2,6~29,16,20R"]
#args = ["GG8", "V0B", "E~-4=0"]
#args = ["GG20W4", "V13R5"]
#args = ["G7", "V5R6"]

#args = ["G12"]
#args = ["G12", "V3::4,-2B"]
#args = ["G12", "V3::4,-2B", "E6,-6,6=2:-4:2R"]
#args = ["G12", "V3::4,-2B", "E6,-6,6=2:-4:2R", "V2:-3:4BR"]
#args = ["G12", "V3::4,-2B", "E6,-6,6=2:-4:2R", "V2:-3:4BR", "V::6BR5"]
#args = ["G12", "V3::4,-2B", "E6,-6,6=2:-4:2R", "V2:-3:4BR", "V::6BR5", "E2::4~-2::-5"]
#args = ["GG24W4R18", "V-19R", "V-22BR7"]
#args = ["GG50W10", "V-26R5"]
#args = ["GG50", "V-37R5"]
SIZE = 12
def defaultWidth(size): #
    sqrtLen = math.sqrt(size)
    length = (int)(sqrtLen)
    while ((size % length != 0)):
        length -= 1
    width = size//length
    return width

def isGraphDirective(arg):
    return arg[0] == "G"

def isVertexDirective(arg):
    return arg[0] == "V"

def isEdgeDirective(arg):
    return arg[0] == "E"

def isNTypeGraphFromArg(arg, width):
    return 'N' in arg or width == 0

def getWidthFromArg(arg, size):
    match = re.search(r'W(\d+)', arg)
    if match:
        return int(match.group(1))
    else:
        return defaultWidth(size)

def getRewardFromArg(arg, defaultReward): #if R[#] is in arg, returns #. if R is in it, returns 12. else returns -1 (no reward)
    if 'R' in arg:
        start_index = arg.find('R') + 1 
        num_str = ''
        while start_index < len(arg) and arg[start_index].isdigit():
            num_str += arg[start_index]
            start_index += 1
        if num_str:
            return int(num_str) 
        else:
            return defaultReward
    #return 12 ######
    return -1


def getVertexFromDirective(directive): #returns the vslice after the "V"
    if 'V' in directive:
        start_index = directive.find('V') + 1 
        num_str = ''
        while start_index < len(directive) and (directive[start_index].isdigit() or directive[start_index] in [":", "-", ","]):
            num_str += directive[start_index]
            start_index += 1
    return num_str

def getSizeFromArg(arg): #returns string of contiguous digits in arg
    numbers = re.findall(r'\d+', arg)
    if numbers:
        return int(numbers[0])
    return None

def expandVslcs(slc_str): #returns array of expanded vslice
    defaultStopForNegativeIncrement = False
    size = SIZE #MAYBE CHANGE THIS 
    result_indices = []
    #split input by commas and process each
    slices = slc_str.split(',')
    for slc in slices:
        slc = slc.strip() 
        if ':' in slc:
            parts = slc.split(':')
            start = int(parts[0]) if parts[0] else None
            stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
            step = int(parts[2]) if len(parts) > 2 and parts[2] else None
            
            #default values
            if start is None:
                start = 0 if step is None or step > 0 else size - 1
            if stop is None:
                if step is None or step > 0:
                    stop = size 
                else:
                    stop = -1
                    defaultStopForNegativeIncrement = True
            if step is None:
                step = 1
            
            #adjust negatives
            if start < 0:
                start += size
            if stop < 0 and not (defaultStopForNegativeIncrement):
                stop += size
            
            #generate the slices
            #print("range(" + str(start) + ", " + str(stop) + ", " + str(step) + ")")
            result_indices += list(range(start, stop, step))
        else: #handle single index
            index = int(slc)
            if index < 0:
                index += size
            result_indices.append(index)
    #adjust (just in case)
    adjusted_indices = [(x % size) for x in result_indices]
    return adjusted_indices

def followDirections(startVertex, cardinalDirections, width, length):
    # Calculate the row and column of the start vertex based on the WIDTH
    WIDTH = width
    LENGTH = length
    row = startVertex // WIDTH 
    startRow = row
    col = startVertex % WIDTH
    startCol = col
    newVertexes = []
    # Process each direction in the cardinalDirections string
    for direction in cardinalDirections:
        row = startRow
        col = startCol
        if direction == 'N': 
            if row == 0:  
                continue
            row -= 1
        elif direction == 'S':
            if row == LENGTH - 1:  
                continue
            row += 1
        elif direction == 'E':  
            if col == WIDTH - 1: 
                continue
            col += 1
        elif direction == 'W':  
            if col == 0:  
                continue
            col -= 1
        newVertex = row * WIDTH + col
        newVertexes.append(newVertex)
    return newVertexes

def createNTypeGraph(size, reward, widthExists):
    adj_list = [[] for _ in range(size)]
    vprops = {i: {} for i in range(size)}
    dict = {
            'size': size,
            'adj_list': adj_list,
            'vprops': vprops
        }
    if reward != -1:
        dict['rwd'] = reward
    if widthExists:
        dict['width'] = 0
    return dict
#returns

def isNTypeGraph(graph):
    return 'width' not in graph or graph['width'] == 0

def createGraph(size, width, length, reward):
    adj_list = [[] for _ in range(size)]
    for i in range(size):
        row = i // width
        col = i % width
        if col > 0:  #connect left 
            adj_list[i].append((i - 1, -1))
        if col < width - 1:  #connect right
            adj_list[i].append((i + 1, -1))
        if row > 0:  #connect up
            adj_list[i].append((i - width, -1))
        if row < length - 1:  #connect down
            adj_list[i].append((i + width, -1))
    vprops = {i: {} for i in range(size)}
    return {
        'size': size,
        'rwd': reward,
        'width': width,
        'length': length,
        'vprops': vprops,
        'adj_list': adj_list
    }

def removeEdge(graph, v1, v2):
    graph['adj_list'][v1] = [(end, weight) for end, weight in graph['adj_list'][v1] if end != v2]
    return graph

def addEdge(graph, v1, v2, rwd):
    graph['adj_list'][v1].append((v2, rwd))
    return graph

def editEdge(graph, startVertex, endVertex, reward):
    for index, (vertex, weight) in enumerate(graph['adj_list'][startVertex]):
        if vertex == endVertex:
            graph['adj_list'][startVertex][index] = (endVertex, reward) ###PROBASBLY WRONG    
    return graph                
def originallyConnected(graph, v1, v2, nType):
    if nType: return False
    width = graph['width']
    return abs(v1 - v2) == width or (abs(v1 - v2) == 1 and v1//width == v2//width) 

def edgeExistsInGraph(graph, v1, v2):
    for vtx, rwd in graph['adj_list'][v1]:
        if vtx == v2:
            return True
    return False

def complementEdge(graph, v1, v2, nType):
    edge_exists = edgeExistsInGraph(graph, v1, v2)
    if edge_exists:
        graph = removeEdge(graph, v1, v2)
        #print(f"Removed edge from {v1} to {v2}") ###########################
    else: #edge doesn't currently existed
        if originallyConnected(graph, v1, v2, nType):
            #print("graph before adding edge:", graph)
            graph = addEdge(graph, v1, v2, -1) #should i do from v2 to v1??/
            #print(f"Added edge from {v1} to {v2} with weight -1") #################
            #print("graph after adding edge:", graph)
    return graph
def seperateEDirectiveType1(arg):
    pattern = r'E([!+*~@])*([:,0123456789-]+)([=~])([:,0123456789-]+)([RT0123456789]*)'
    match = re.match(pattern, arg)
    management = match.group(1)
    if management == None:
        management = "~"
    vslcs1 = match.group(2)
    edge_type_char = match.group(3)
    vslcs2 = match.group(4)
    properties = match.group(5)
    return management, vslcs1, edge_type_char, vslcs2, properties
def seperateEDirectiveType2(arg):
    pattern = r'E([!+*~@])*([:,0123456789-]+)([NSEW]+)([=~])*([RT0123456789]*)'
    match = re.match(pattern, arg)
    management = match.group(1)
    if management == None:
        management = "~"
    vslcs1 = match.group(2)
    directions = match.group(3)
    edge_type_char = match.group(4)
    properties = match.group(5)
    return management, vslcs1, directions, edge_type_char, properties

def editGraph(graph, management, startVertex, endVertex, reward):
    edge_exists = edgeExistsInGraph(graph, startVertex, endVertex)
    if management == "!":
        graph = removeEdge(graph, startVertex, endVertex)
    elif management == "+":
        if not edge_exists:
            graph = addEdge(graph, startVertex, endVertex, reward)
    elif management == "*":
        if not edge_exists:
            graph = addEdge(graph, startVertex, endVertex, reward)
        else:
            graph = editEdge(graph, startVertex, endVertex, reward)
    elif management == "~":
        #print("HERE")
        if not edge_exists:
            #print("addEdge()", startVertex, endVertex, reward)
            graph = addEdge(graph, startVertex, endVertex, reward)
        else:
            graph = removeEdge(graph, startVertex, endVertex)
    elif management == "@":
        if edge_exists:
            graph = editEdge(graph, startVertex, endVertex, reward)
    return graph
#returns a graph object
def grfParse(lstArgs): 
    graph = None
    nType = False
    for arg in lstArgs:
        if isGraphDirective(arg):
            size = getSizeFromArg(arg)
            global SIZE
            global WIDTH
            global SIZE
            SIZE = size
            defaultReward = getRewardFromArg(arg, 12)
            if defaultReward == -1:
                defaultReward = 12
            WIDTH = getWidthFromArg(arg, size)
            if isNTypeGraphFromArg(arg, WIDTH):
                widthExists = False
                if WIDTH == 0:
                    widthExists = True
                graph = createNTypeGraph(size, defaultReward, widthExists)
                nType = True
                continue
            LENGTH = size//WIDTH
            graph = createGraph(size, WIDTH, LENGTH, defaultReward)
        elif isVertexDirective(arg):
            vSlice = getVertexFromDirective(arg)
            vertices = expandVslcs(vSlice)
            vertices = list(dict.fromkeys(vertices)) #remove duplicates
            #print("vertices:", vertices)
            all_vertices = list(range(size))
            non_vertices = [node for node in all_vertices if node not in vertices]
            #print("non vertices:", non_vertices)
            if 'B' in arg:
                for v1 in vertices:
                    for v2 in non_vertices:
                        graph = complementEdge(graph, v1, v2, nType)
                        graph = complementEdge(graph, v2, v1, nType) ###
            if 'R' in arg:
                r = getRewardFromArg(arg, defaultReward)
                for v in vertices:
                    graph['vprops'][v]["rwd"] = r
            
        elif isEdgeDirective(arg):
            if 'N' in arg or 'S' in arg or 'W' in arg or 'E' in arg[1:]: #Type 2
                modifyEdgeList = []
                #generate the modifyEdgeList
                management, vslcs1, directions, edge_type_char, properties = seperateEDirectiveType2(arg)
                reward = getRewardFromArg(properties, defaultReward)
                startVertexList = expandVslcs(vslcs1)
                vertex_list = expandVslcs(vslcs1)
                for startVertex in vertex_list: #
                    endVertexes = followDirections(startVertex, directions, WIDTH, LENGTH)
                    for endVertex in endVertexes:
                        if endVertex == -1: continue
                        if edge_type_char == "=":
                            modifyEdgeList.append((startVertex, endVertex))
                            modifyEdgeList.append((endVertex, startVertex))
                        elif edge_type_char == "~":
                            modifyEdgeList.append((startVertex, endVertex))
                        else:
                            print("ERROR!!!!")
                for edge in modifyEdgeList:
                    startVertex = edge[0]
                    endVertex = edge[1]
                    graph = editGraph(graph, management, startVertex, endVertex, reward)
            else: #Type 1
                modifyEdgeList = []
                management, vslcs1, edge_type_char, vslcs2, properties = seperateEDirectiveType1(arg)
                reward = getRewardFromArg(properties, defaultReward)
                startVertexList = expandVslcs(vslcs1)
                endVertexList = expandVslcs(vslcs2) 
                modifyEdgeList = list(zip(startVertexList, endVertexList))
                if edge_type_char == "=":
                    moreEdges = list(zip(endVertexList, startVertexList))
                    modifyEdgeList.extend(moreEdges)
                modifyEdgeList = list(dict.fromkeys(modifyEdgeList))
                #print(modifyEdgeList)
                for edge in modifyEdgeList:
                    
                    startVertex = edge[0]
                    endVertex = edge[1]
                    graph = editGraph(graph, management, startVertex, endVertex, reward)
    return graph

#returns the size of the graph (number of vertices)
def grfSize(graph): 
    return graph['size']
    pass

#returns a dictionary of the graph properties, which always includes the default 'rwd' and 'width' for gridworld type of graphs.
def grfGProps(graph):
    dict = {}
    if 'width' in graph: 
        dict['width'] = graph['width']
    if 'rwd' in graph:
        dict['rwd'] = graph['rwd']
    return dict


#returns an iterator (list, set, dictionary) that yields the neighbors of vertx vtx
def grfNbrs(graph, vtx): 
    edgeList = graph['adj_list'][vtx] #edge list for this vertex
    return [edge[0] for edge in edgeList]
 
#returns a dictionary of the vertex properties of vertex vtx
def grfVProps(graph, vtx): 
    # vtx_properties = {}
    # if vtx not in graph['adj_list']:
    #     return vtx_properties
    # for endVertex, reward in graph['adj_list'][vtx]:
    #     if reward != -1:
    #         vtx_properties[endVertex] = reward
    
    # return {"rwd": vtx_properties}
    return graph['vprops'][vtx]

#returns a dictionary of the edge properties of edge (v1, v2)
def grfEProps(graph, v1, v2): 
    for (vertex, reward) in graph['adj_list'][v1]:
        # Check if the current edge is the one to v2
        if vertex == v2 and reward != -1:
            return {"rwd": reward}
    return {}
    pass



def cardinalNodes(graph, current, neighbor): 
    if isNTypeGraph(graph):
        return False
    width = graph['width']
    if (neighbor == current - 1 and current % width != 0):
        return True
    if (neighbor == current + 1 and current % width != width-1):
        return True
    if neighbor == current - width:
        return True
    if neighbor == current + width:
        return True
    return False
def getJumps(graph):
    jumpEdges = []
    size = graph['size']
    for current in range(size):
        current_neighbors = graph['adj_list'][current]
        for neighbor, rwd in current_neighbors:
            if not cardinalNodes(graph, current, neighbor):
                jumpEdges.append(str(current) + "~" + str(neighbor))
    if len(jumpEdges) == 0:
        return ""
    else:
        return "Jumps: " + ";".join(jumpEdges)
#returns the 1D grid representation of the graph (a string of length grfSize()), followed by the jumps
def grfStrEdges(graph):  #optimization could be to do it once instead of twice
    if 'width' in graph:
        width = graph['width']
    def getDirection(current, neighbor):
        if neighbor == current - 1 and current % width != 0:
            return 'W'
        elif neighbor == current + 1 and current % width != width-1:
            return 'E'
        elif neighbor == current - width:
            return 'N'
        elif neighbor == current + width:
            return 'S'
        return ''
    size = graph['size']
    edge_encoding = []
    for i in range(size):
        current_neighbors = graph['adj_list'][i]
        #print("current_neighbors:", current_neighbors)
        directions = ''.join(sorted(getDirection(i, nbr[0]) for nbr in current_neighbors))
        direction_map = {
            '': '.',   
            'N': 'N', 'S': 'S', 'E': 'E', 'W': 'W',
            'NS': '|', 'EW': '-',
            'EN': 'L', 'ES': 'r', 'SW': '7', 'NW': 'J',
            'ENSW': '+',
            'ENW': '^', 'ESW': 'v',
            'ENS': '>', 'NSW': '<'
        }
        #print(directions)
        encoded_char = direction_map.get(directions, "")
        edge_encoding.append(encoded_char)
        #print("Jumps:", ) ************
    

    toRet = ''.join(edge_encoding)
    if isNTypeGraph(graph):
        toRet = ""
    #if 'width' in graph and width != 0:
        #toRet = '\n'.join(''.join(edge_encoding[i * width:(i + 1) * width]) for i in range(size // width))
    jumpsStr = getJumps(graph)
    # if len(edgeJumps) > 0:
    #     toRet += "; Jumps: " + str(edgeJumps)
    if jumpsStr:
        toRet += ("\n" + jumpsStr)
    #print("toRet")
    #print(toRet)
    return toRet

#returns the string representation of each graph, vertex, and edge property
def grfStrProps(graph):
    grfProps = {key: value for key, value in graph.items() if key != "adj_list" and key != "size" and key != 'vprops' and key != "length" and not(key == 'rwd' and value == -1)}
    output = ""
    for vertex, properties in graph['vprops'].items():
        if properties:
            output += f"{vertex}: {properties}\n"

    for startVertex, edges in enumerate(graph['adj_list']):
        for endVertex, reward in edges:
            if reward != -1:
                output += f"({startVertex},{endVertex}):'rwd': '{reward}'\n"
    # print("output", output)
    # print("STARTTTT")
    # print(str(grfProps) + "\n" + output)
    # print("END")
    #***** ADD THE OTHER PROPERTIES
    return str(grfProps) + "\n" + output
    #pass

def formatEdgesStr(edgesStr, size, width):
    lines = []
    for i in range(0, size, width):
        line = edgesStr[i:i + width]
        lines.append(line)
    formatted_string = '\n'.join(lines)
    if len(edgesStr) > size:
        formatted_string += edgesStr[size:]
    return formatted_string

def formatPolicyStr(directions, width):
    result = ""
    for index in range(len(directions)):
        result += directions[index]
        if (index + 1) % width == 0:
            result += "\n"
    return result

def findShortestDistanceToReward(graph, start):
    visited = [False] * graph['size']
    queue = [(start, 0)]  #(current node, current distance)
    visited[start] = True
    qIndex = 0 #pointer to current index in queue

    while qIndex < len(queue):
        current, dist = queue[qIndex]
        qIndex += 1
        #does current node have reward?
        if 'rwd' in graph['vprops'][current]:
            return dist

        #explore negihbors
        for neighbor, reward in graph['adj_list'][current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                if reward > 0:  
                    return dist + 1 
                queue.append((neighbor, dist + 1))

    return -1 
def determineInitialDirections(graph, width, size):
    height = size//width
    directions = {}
    for i in range(graph['size']):
        if 'rwd' in graph['vprops'][i]:
            directions[i] = '*'
            continue
        min_distance = float('inf')
        initial_moves = set()
        #check 4 cardinal directions
        for dx, dy, dir_code in [(0, -1, 'N'), (0, 1, 'S'), (1, 0, 'E'), (-1, 0, 'W')]:
            x, y = i % width, i // width
            nx, ny = x + dx, y + dy
            neighbor = ny*width + nx
            if not edgeExistsInGraph(graph, i, neighbor):
                continue
            if 0 <= nx < width and 0 <= ny < height:
                if neighbor < graph['size']:
                    dist = findShortestDistanceToReward(graph, neighbor)
                    if dist != -1:
                        dist += 1  #Adding the edge from i to neighbor
                        if dist < min_distance:
                            min_distance = dist
                            initial_moves = {dir_code}
                        elif dist == min_distance:
                            initial_moves.add(dir_code)

        if min_distance == float('inf'):
            directions[i] = '.'
        else:
            dir_str = ''.join(sorted(initial_moves))
            dir_map = {'NS': '|', 'EW': '-', 'EN': 'L', 'ES': 'r', 'SW': '7', 'NW': 'J',
                       'ENSW': '+', 'ENW': '^', 'ESW': 'v', 'NES': '>', 'NSW': '<'}
            directions[i] = dir_map.get(dir_str, dir_str) if len(dir_str) > 1 else dir_str

    return directions

def main():
    graph = grfParse(args)
    #print(graph)
    #print(grfGProps(graph))
    #print(grfNbrs(graph, 1))
    #print("SIZE:", SIZE)
    edgesStr = grfStrEdges(graph)
    propsStr = grfStrProps(graph)
    #jumpsStr = getJumps(graph)
    if 'width' in graph:
        width = graph['width']
        size = graph['size']
    if not isNTypeGraph(graph):
        edgesStr = formatEdgesStr(edgesStr, size, width)
    print(edgesStr)
    #if jumpsStr: print(jumpsStr)
    
    print(propsStr)
    #directions = determineInitialDirections(graph, graph['width'], graph['size'])
    #print("GRAPH:", graph)
    #print("Policy:\n", formatPolicyStr(directions, graph['width']))
    #print(grfNbrs(graph,2))
    #print(grfNbrs(graph,7))
    #print(grfNbrs(graph,6))
    #print(grfNbrs(graph,5)) ###
    #print(grfNbrs(graph,9))
    #print(grfVProps(graph, 9))
    #print(grfEProps(graph,2,16))
    #print(graph)

    # for vertex, properties in graph['vprops'].items():
    #     if properties:
    #         print(f"{vertex}: {properties}")
    # for startVertex, edges in enumerate(graph['adj_list']):
    #     for endVertex, reward in edges:
    #         # Check if the reward is not -1
    #         if reward != -1:
    #             # Output the edge and reward
    #             print(f"({startVertex},{endVertex}):'rwd': '{reward}'")
    # print("START:")
    # print(grfStrProps(graph))

if __name__ == '__main__': main()

#andrew chen pd6 2026
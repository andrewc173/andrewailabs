#NEURAL NETWORKS LAB, MOVED TO NEW FOLDER
import sys; args = sys.argv[1:]
import re
import math
args = ['weights/weights5222.txt', 'T4', '0.0', '1.4', '-0.4', '0.0', '-0.5']

def dot_product(v1, v2): #dot product of vectors v1 and v2
    return sum([v1[i]*v2[i] for i in range(len(v1))])

def t1(x): #for a linear function, f(x)=x
    return x

def t2(x): #for a ramp function (f(x)=x for x>=0)
    return max(0, x)

def t3(x): #for a logistic function (f(x)=1/(1+e^-x))
    return 1/(1+math.exp(-x))

def t4(x): #for twice the T3 logistic function less 1.
    return 2*t3(x) - 1

def parse_inputs(lstArgs): 
    #example of lstArgs: ['weights/weights101.txt', 'T3', '1.3']
    #another example: ['weights/weights5222.txt', 'T4', '0.0', '1.4', '-0.4', '0.0', '-0.5']
    #inputs is the numnbers at the end of the list
    inputs = []
    for arg in lstArgs: 
        if re.match(r'weights\d+.txt', arg):
            file = arg
            openFile = open(file, 'r')
            weights = openFile.read().splitlines()
        elif re.match(r'T\d', arg):
            function = arg
        elif re.match(r'-?\d+(\.\d+)?', arg):
            inputs.append(float(arg))   
    print(weights, function, inputs)
    return weights, function, inputs

def main():
    weights, function, inputs = parse_inputs(args)
    print(weights)
    print(function)
    print(inputs)

if __name__ == '__main__': main()


#andrew chen pd6 2026
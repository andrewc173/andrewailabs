import sys; args = sys.argv[1:]
import re
import math
#args = ['weights/weights244.txt', 'T2', '1.7', '-0.2']
#args = ['weights/weights5222.txt', 'T4', '1.6', '0.6', '-2.0', '1.7', '2.0']
#args = ['weights/weights101.txt', 'T3', '-1.9']
#args = ['weights/weights69.txt', 'T2', '2.0', '1.1']

def dot_product(v1, v2): #dot product of vectors v1 and v2
    dot_product_str = " + ".join([f"{v1[i]}*{v2[i]}" for i in range(len(v1))])
    dot_product_str = f"{dot_product_str} = {sum([v1[i]*v2[i] for i in range(len(v1))])}"
    #print(dot_product_str)
    return sum([v1[i]*v2[i] for i in range(len(v1))])

def dot_product_not_summed(v1, v2): #dot product of vectors v1 and v2
    dot_product_str = " + ".join([f"{v1[i]}*{v2[i]}" for i in range(len(v1))])
    dot_product_str = f"{dot_product_str} = {sum([v1[i]*v2[i] for i in range(len(v1))])}"
    #print(dot_product_str)
    return [v1[i]*v2[i] for i in range(len(v1))]

def t1(x): #for a linear function, f(x)=x
    return x

def t2(x): #for a ramp function (f(x)=x for x>=0)
    return max(0, x)

def t3(x): #for a logistic function (f(x)=1/(1+e^-x))
    return 1/(1+math.exp(-x))

def t4(x): #for twice the T3 logistic function less 1.
    return 2*t3(x) - 1

def parse_inputs(args): 
    #example of args ['weights/weights101.txt', 'T3', '1.3']
    #another example: ['weights/weights5222.txt', 'T4', '0.0', '1.4', '-0.4', '0.0', '-0.5']
    inputs = []
    weights = open(args[0]).read().split("\n")
    print(weights)
    weights = [list(map(float, weight.split())) for weight in weights if weight != ""]
    t_function = int(args[1][1:])
    inputs = [float(arg) for arg in args[2:]]
    return weights, t_function, inputs

def produceNewInputs(weights, inputs, last):
    #example of weights: 0.6 0.8 0.4 -0.4 0.2 0.9 -0.3 0.4 0.9 0.5
    #example of inputs: 0.0 1.4 -0.4 0.0 -0.5
    new_inputs = []
    weights = [weights[i:i+len(inputs)] for i in range(0, len(weights), len(inputs))]
    for weight in weights:
        # if len(weight) == 0:
        #     new_inputs.append(0)

        if last:
            new_inputs.append(dot_product_not_summed(weight, inputs))
        else:
            new_inputs.append(dot_product(weight, inputs))
    # print("NEW+inputs:", new_inputs)
    return new_inputs
def doNeuralNetwork(allWeights, t_function, inputs):
    new_inputs = inputs
    for i, weights in enumerate(allWeights):
        last = i == len(allWeights) - 1
        #print("Does neural network on \ninput:", new_inputs, "\nweights:", weights)
        new_inputs = produceNewInputs(weights, new_inputs, last)
        #print("result:", new_inputs)
        if i == len(allWeights) - 1:
            break
        if t_function == 1:
            new_inputs = [t1(x) for x in new_inputs]
        elif t_function == 2:
            new_inputs = [t2(x) for x in new_inputs]
        elif t_function == 3:
            new_inputs = [t3(x) for x in new_inputs]
        elif t_function == 4:
            new_inputs = [t4(x) for x in new_inputs]
        #print("after t function:", new_inputs)
        #print("\n")
        #print("new inputs:", new_inputs)
    return new_inputs

def main():
    allWeights, t_function, inputs = parse_inputs(args)
    #print("ARGUMENTS")
    #print("all weights:", allWeights)
    #print("function:", t_function)
    #print("inputs", inputs)
    #print("====================================")
    output = doNeuralNetwork(allWeights, t_function, inputs)
    #output = [round(x,3) for x in output]
    output = str(output)
    output = output[1:len(output)-1]
    print(output)
    

if __name__ == '__main__': main()


#andrew chen pd6 2026
import sys; args = sys.argv[1:]
#infile = open(args[0], 'r')
#infile = open(args[0])
#infile = open('test.txt')
#infile = open('x_gate.txt')
infile = open('x_gate_2.txt')
#infile = open('x_gate_3.txt')
#infile = open('test11.txt')
import math, random

# t_funct is symbol of transfer functions: 'T1', 'T2', 'T3', or 'T4'
# input is a list of input (summation) values of the current layer
# returns a list of output values of the current layer
def transfer(t_funct, input):
   if t_funct == 'T3': return [1 / (1 + math.e**-x) for x in input]
   elif t_funct == 'T4': return [-1+2/(1+math.e**-x) for x in input]
   elif t_funct == 'T2': return [x if x > 0 else 0 for x in input]
   else: return [x for x in input]

# returns a list of dot_product result. the len of the list == stage
# dot_product([x1, x2, x3], [w11, w21, w31, w12, w22, w32], 2) => [x1*w11 + x2*w21 + x3*w31, x1*w12, x2*w22, x3*w32] 
def dot_product(input, weights, stage):
   return [sum([input[x]*weights[x+s*len(input)] for x in range(len(input))]) for s in range(stage)]

def dot_product_for_last (input, weights, stage):
  temporary = []
  for i in range(stage):
    temporary.append(input[i] * weights[i])
  return temporary

# Complete the whole forward feeding for one input(training) set
# return updated x_vals and error of the one forward feeding
def ff(ts, xv, weights, t_funct):
   ''' ff coding goes here '''
   for i in range(len(weights) - 1):
      xv[i + 1] = dot_product(xv[i], weights[i], len(xv[i + 1]))
      if i < len(weights) - 1:
         xv[i + 1] = transfer(t_funct, xv[i + 1])
   xv[len(weights)] = dot_product_for_last(xv[len(weights) - 1], weights[-1], len(weights[-1]))

   err = sum([(ts[i-len(xv[-1])] - xv[-1][i])**2 for i in range(len(xv[-1]))]) / 2
   return xv, err

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return z*(1-z)

# Complete the back propagation with one training set and corresponding x_vals and weights
# update E_vals (ev) and negative_grad, and then return those two lists


def bp(ts, xv, weights, ev, negative_grad):   
   ''' bp coding goes here '''
   for i in range(len(ev[-1])):
      ev[-1][i] = ts[i - len(ev[-1])] - xv[-1][i]
      negative_grad[-1][i] = ev[-1][i] * xv[-2][i]
   for i in range(len(ev[-2])):
      ev[-2][i] = weights[-1][i] * ev[-1][i] * sigmoid_prime(xv[-2][i])
      
   # negative gradient
   index = 0
   for j in range(len(ev[-2])):
      for i in range(len(ev[-3])):    
         negative_grad[-2][index] = xv[-3][i] * ev[-2][j]
         index += 1
   # e-values   
   for i in range(len(ev[-3])):
      value = 0 # weight multiplied by the e-value
      for j in range(len(ev[-2])):
         value += weights[-2][j * len(ev[-3]) + i] * ev[-2][j]
      #ev[-3][i] = value * xv[-3][i] * (1 - xv[-3][i])
      ev[-3][i] = value * sigmoid_prime(xv[-3][i])
      #ev[-3][i] = value
   # negative gradient
   index = 0
   for j in range(len(ev[-3])):
      for i in range(len(ev[-4])):  
         negative_grad[-3][index] = xv[-4][i] * ev[-3][j]
         index += 1

   bp_calculate(ev, weights, 'T3', xv, negative_grad, -3)      
   return ev, negative_grad

def bp_calculate(ev, weights, t_funct, xv, negative_grad, layer):
    # e-values
    for i in range(len(ev[layer])):
        value = 0 # weight multiplied by the e-value
        for j in range(len(ev[layer + 1])):
            value += weights[layer + 1][j * len(ev[layer]) + i] * ev[layer + 1][j]
        ev[layer][i] = value * sigmoid_prime(xv[layer][i])
    # negative gradient
    index = 0
    for j in range(len(ev[layer])):
        for i in range(len(ev[layer - 1])):
            negative_grad[layer][index] = xv[layer - 1][i] * ev[layer][j]
            index += 1

# update all weights and return the new weights
# Challenge: one line solution is possible
def update_weights(weights, negative_grad, alpha):
   ''' update weights (modify NN) code goes here '''
   for i in range(len(weights)):
      for j in range(len(weights[i])):
         weights[i][j] = weights[i][j] + negative_grad[i][j] * alpha
   return weights

def main():
   t_funct = 'T3' # we default the transfer(activation) function as 1 / (1 + math.e**(-x))
   ''' work on training_set and layer_count '''
   lines = infile.readlines()
   infile.close()
   training_set = []  # list of lists
   for line in lines:
      components = line.split()
      arrow_index = components.index("=>")
      training_set.append([float(x) for x in components[:arrow_index]] + [float(x) for x in components[arrow_index + 1:]])
   num_inputs = len([float(x) for x in components[:arrow_index]])
   num_outputs = len([float(x) for x in components[arrow_index + 1:]])
   #print ("TRAINING SET", training_set) #[[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]]
   layer_counts = [num_inputs + 1, num_outputs + 1, num_outputs, num_outputs] # set the number of layers: (# of input + 1), (# of output + 1), # of output, # of output
   print ("LAYER COUNT", layer_counts) # This is the first output. [3, 2, 1, 1] with the given x_gate.txt
   
   ''' build NN: x nodes and weights '''
   x_vals = [[temp[0:num_inputs]] for temp in training_set] # x_vals starts with first input values
   #print (x_vals) # [[[1.0, -1.0]], [[-1.0, 1.0]], [[1.0, 1.0]], [[-1.0, -1.0]], [[0.0, 0.0]]]
   # make the x value structure of the NN by putting bias and initial value 0s.
   for i in range(len(training_set)):
      for j in range(len(layer_counts)):
         if j == 0: x_vals[i][j].append(1.0)
         else: x_vals[i].append([0 for temp in range(layer_counts[j])])
   #print ("X_VALUES", x_vals) # [[[1.0, -1.0, 1.0], [0, 0], [0], [0]], [[-1.0, 1.0, 1.0], [0, 0], [0], [0]], ...

   # by using the layer counts, set initial weights [3, 2, 1, 1] => 3*2 + 2*1 + 1*1: Total 6, 2, and 1 weights are needed
   weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])]  for i in range(len(layer_counts)-2)]
   weights.append([round(random.uniform(-2.0, 2.0), 2) for i in range(layer_counts[-1])])
   #weights = [[1.35, -1.34, -1.66, -0.55], [-1.08, -0.7], [-0.6]]   # For 1 => 1 0 => 0 
   #weights = [[0.22, -0.15, 1.95, 1.86, -1.05, 1.66, -1.68, 1.05, -1.16, 0.38, 0.67, 0.23], [-1.87, 1.87, 0.46, 0.28, -0.07, 1.88], [1.48, -1.07]]  #Alina 3
   #weights = [[-1.76, 0.35, 1.98, 0.79, 1.63, -1.11, -0.61, 0.33, 0.19, -0.15, -0.57, 1.15], [-1.54, 1.71, 1.55, -1.26, -0.76, 1.45], [1.65, 1.52]]  #Eric Example 3
   #weights = [[1.35, -1.34, -1.66, -0.55, -0.9, -0.58, -1.0, 1.78], [-1.08, -0.7], [-0.6]]   #Example 2
   #print ("WEIGHTS", weights)    #[[2.0274715389784507e-05, -3.9375970265443985, 2.4827119599531016, 0.00014994269071843774, -3.6634876683142332, -1.9655046461270405]
                        #[-3.7349985848630634, 3.5846029322774617]
                        #[2.98900741942973]]

   # build the structure of BP NN: E nodes and negative_gradients 
   E_vals = [[[*i] for i in j] for j in x_vals]  #copy elements from x_vals, E_vals has the same structures with x_vals
   negative_grad = [[*i] for i in weights]  #copy elements from weights, negative gradients has the same structures with weights
   errors = [10]*len(training_set)  # Whenever FF is done once, error will be updated. Start with 10 (a big num)
   count = 1  # count how many times you trained the network, this can be used for index calc or for decision making of 'restart'
   alpha = 0.3
   
#    print ("TRAINING SET", training_set)
#    print ("LAYER COUNT", layer_counts)
#    print ("X_VALUES", x_vals)
#    print ("WEIGHTS", weights)
#    print("E-VALUES", E_vals)
#    print("NEGATIVE GRADIENT", negative_grad)
#    print("ERRORS", errors)
   # calculate the initial error sum. After each forward feeding (# of training sets), calculate the error and store at error list
   for k in range(len(training_set)):
      x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
      #sum??
      E_vals[k], negative_grad = bp(training_set[k], x_vals[k], weights, E_vals[k], negative_grad)
      #modify weights
      weights = update_weights(weights, negative_grad, alpha)
   err = sum(errors)
#    print("----------DIVIDER----------")
   
#    print ("TRAINING SET", training_set)
#    print ("LAYER COUNT", layer_counts)
#    print ("X_VALUES", x_vals)
#    print ("WEIGHTS", weights)
#    print("E-VALUES", E_vals)
#    print("NEGATIVE GRADIENT", negative_grad)
#    print("ERRORS", errors)
#    print(err)
   ''' 
   while err is too big, reset all weights as random values and re-calculate the error sum.
   
   '''
   while err > 5:
      weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])]  for i in range(len(layer_counts)-2)]
      weights.append([round(random.uniform(-2.0, 2.0), 2) for i in range(layer_counts[-1])])
      for k in range(len(training_set)):
        x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
      err = sum(errors)

   '''
   while err does not reach to the goal and count is not too big,
      update x_vals and errors by calling ff()
      whenever all training sets are forward fed, 
         check error sum and change alpha or reset weights if it's needed
      update E_vals and negative_grad by calling bp()
      update weights
      count++
   '''
   
   while err > 0.01:
      for k in range(len(training_set)):
        x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
        E_vals[k], negative_grad = bp(training_set[k], x_vals[k], weights, E_vals[k], negative_grad)
        weights = update_weights(weights, negative_grad, alpha)
      count += 1
      err = sum(errors) 
      if err > 3 or count > 2000000:
        alpha = random.uniform(0, 1)
 #       print('count', count)
        weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])]  for i in range(len(layer_counts)-2)]
        weights.append([round(random.uniform(-2.0, 2.0), 2) for i in range(layer_counts[-1])])
        count = 1   
#      print("Error", err)
  
   print("Error", err)
   # print final weights of the working NN
   print ('Weights:')
   for w in weights: print (w)
if __name__ == '__main__': main()

# python3 NeuralNetwork.py x_gate_2.txt
# Alina Chen, 5, 2024
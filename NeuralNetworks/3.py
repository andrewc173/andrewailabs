import sys; args = sys.argv[1:]
#args = ["x_gate_2.txt"]
#args = ["x_gate_3.txt"]
#infile = open(args[0], 'r')
infile = open(args[0])

import math, random

def transfer_function(t_funct, input):
   if t_funct == 'T3': return [1 / (1 + math.e**-x) for x in input]
   elif t_funct == 'T4': return [-1+2/(1+math.e**-x) for x in input]
   elif t_funct == 'T2': return [x if x > 0 else 0 for x in input]
   else: return [x for x in input]

def sigmoid_dx(x):
    return x*(1-x)

def backward_propagation(expected_result, x_values, weights, err_vals, negative_grad):   
    
    for layer in range(len(x_values) - 2, -1, -1):
        for i in range(len(x_values[layer])):
            if layer == len(x_values) - 2: # Last Layer
                #e_3
                err_vals[layer + 1][i] = expected_result[i] - x_values[layer +1][i]  
                #g_2
                negative_grad[layer][i] = x_values[layer][i] * err_vals[layer + 1][i]
                        
                for i in range(len(err_vals[layer])):
                    #e_2
                    err_vals[layer][i] = weights[layer][i] * err_vals[layer + 1][i] * sigmoid_dx(x_values[layer][i])

            else:
                index = 0
                for j in range(len(err_vals[layer + 1])):
                    for k in range(len(x_values[layer])):    
                        negative_grad[layer][index] = x_values[layer][k] * err_vals[layer + 1][j]
                        index += 1

                for k in range(len(err_vals[layer])):
                    sum_errors = 0 
                    for j in range(len(err_vals[layer + 1])):
                        sum_errors += weights[layer][j * len(err_vals[layer]) + k] * err_vals[layer + 1][j]
                    err_vals[layer][k] = sum_errors * sigmoid_dx(x_values[layer][k])

    return err_vals, negative_grad

def update_weights(weights, negative_grad, alpha):
   for i in range(len(weights)):
      for j in range(len(weights[i])):
         weights[i][j] = weights[i][j] + negative_grad[i][j] * alpha
   return weights

def next_value(input, weights, len_next_x_values):
   return [sum([input[x]*weights[x+s*len(input)] for x in range(len(input))]) for s in range(len_next_x_values)]

def forward_propagation(expected_result, x_values, weights, transfer_funct):
   for layer in range(len(x_values) - 2):
      x_values[layer + 1] = next_value(x_values[layer], weights[layer], len(x_values[layer + 1]))
      
      if layer < len(x_values) - 2:
         x_values[layer + 1] = transfer_function(transfer_funct, x_values[layer + 1])

# last value
   x_values[-1] = [x_values[-2][i] * weights[-1][i] for i in range(len(weights[-1]))]

   #err = sum([(expected_result[i-len(x_values[-1])] - x_values[-1][i])**2 for i in range(len(x_values[-1]))]) / 2
   err = sum([(expected_result[i] - x_values[-1][i])**2 for i in range(len(x_values[-1]))]) / 2
   return x_values, err

def main():
   t_funct = 'T3' 
   epochs = 100000
   epoch = 0  
   alpha = 0.1
   sum_err = 10

   lines = infile.readlines()
   infile.close()
   trains = []  # list of lists
   expected_results = []

   for line in lines:
      splits = line.split()
      arrow_position = splits.index("=>")
      trains.append([float(x) for x in splits[:arrow_position]] + [float(x) for x in splits[arrow_position + 1:]])
      expected_results.append([float(x) for x in splits[arrow_position + 1:]])

   inputs_count = len([float(x) for x in splits[:arrow_position]])
   outputs_count = len([float(x) for x in splits[arrow_position + 1:]])
   
   layer_counts = [inputs_count + 1, outputs_count + 1, outputs_count, outputs_count]
   print ("LAYER COUNT", layer_counts) 
   
   x_vals = [[temp[0:inputs_count]] for temp in trains] 
   
   for i in range(len(trains)):
      for j in range(len(layer_counts)):
         if j == 0: x_vals[i][j].append(1.0)
         else: x_vals[i].append([0 for temp in range(layer_counts[j])])

   weights = [[random.random() for j in range(layer_counts[i]*layer_counts[i+1])]  for i in range(len(layer_counts)-2)]
   weights.append([random.random() for i in range(layer_counts[-1])])
   
   # build the structure of BP NN: E nodes and negative_gradients 
   err_vals = [[[*i] for i in j] for j in x_vals]  # deep copy of x_values
   negative_grad = [[*i] for i in weights]  #copy elements from weights, negative gradients has the same structures with weights
   errors = [10]*len(trains)  # Whenever FF is done once, error will be updated. Start with 10 (a big num)

   while sum_err >= 0.0099:
      for k in range(len(trains)):
        x_vals[k], errors[k] = forward_propagation(expected_results[k], x_vals[k], weights, t_funct)
        err_vals[k], negative_grad = backward_propagation(expected_results[k], x_vals[k], weights, err_vals[k], negative_grad)
        weights = update_weights(weights, negative_grad, alpha)
      epoch += 1
      sum_err = sum(errors) 
      if epoch > epochs:
        break
  
   print("Error", sum_err)
   print ('Weights:')
   for w in weights: print (w)

if __name__ == '__main__': main()

# Andrew Chen, pd6, 2026
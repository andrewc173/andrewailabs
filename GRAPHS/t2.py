import re

input_string = "E7N="
pattern = r'E([!+*~@])*([:,0123456789-]+)([NSEW]+)([=~])*([RT0123456789]*)'
match = re.match(pattern, input_string)
management = match.group(1)
vslcs1 = match.group(2)
directions = match.group(3)
edge_type = match.group(4)
rate = match.group(5)

print(management)
print(vslcs1)
print(directions)
print(edge_type)
print(rate)
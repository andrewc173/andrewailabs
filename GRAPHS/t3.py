import re

arg = "E~0=1:3"
pattern = r'E([!+*~@])*([:,0123456789-]+)([=~])([:,0123456789-]+)([RT0123456789]*)'
modifyEdgeList = []
match = re.match(pattern, arg)
management = match.group(1)
vslcs1 = match.group(2)
edge_type_char = match.group(3)
vslcs2 = match.group(4)
rate = match.group(5)

print(management)
print(vslcs1)
print(edge_type_char)
print(vslcs2)
print(rate)
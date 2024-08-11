#testing vslices method
def expandVslcs(slc_str):
    defaultStopForNegativeIncrement = False
    size = 12
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
            print(slc)
            index = int(slc)
            if index < 0:
                index += size
            result_indices.append(index)
    #adjust (just in case)
    #print(range(9,13,-9))
    adjusted_indices = [(x % size) for x in result_indices]
    return adjusted_indices

#print(expandVslcs("-3"))
#print(expandVslcs("1::2"))
#print(expandVslcs("-2::-3, 3::4"))
#print(expandVslcs("1,-2,3"))
#print(expandVslcs("1:4, 6:9, -2:"))
#print(expandVslcs("1::3, 5::3, -2:"))
#print(expandVslcs("3:5"))
#print(expandVslcs("2:7:2"))
#print(expandVslcs("8::-1"))
print(expandVslcs("-10,10"))
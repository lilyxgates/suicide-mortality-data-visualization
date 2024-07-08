#Running Mean function June 9, 2017

import numpy as np  # Rearranged so that import numpy is preceding the 

def running_mean(in_array,window):
        w = int(np.floor(window/2))
        mean_array = np.zeros(len(in_array),dtype=np.float64)
        i = int(w)
        while i < len(in_array)-w:
                mean_array[i] = np.mean(in_array[(i-w):(i+w+1)])
                i += 1
        return(mean_array)


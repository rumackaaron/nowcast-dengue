import numpy as np

# Epiweeks and c_values have the same length
# c_values[i] is cumulative value at epiweek epiweeks[i]
def cum_to_week(epiweeks, c_values):
    epiweeks = np.array(epiweeks)
    c_values = np.array(c_values)
    if len(epiweeks) != len(c_values):
        raise Exception("epiweeks and c_values must have same length")

    idxs = np.argsort(epiweeks)
    epiweeks = epiweeks[idxs]
    c_values = c_values[idxs]
    
    result = np.zeros((len(idxs)))
    prev_yr = -1
    for i in range(len(result)):
        if epiweeks[i] // 100 != prev_yr:
            result[i] = c_values[i]
            prev_yr = epiweeks[i] // 100
        else:
            result[i] = c_values[i] - c_values[i-1]
    return result

import numpy as np

import delphi.utils.epiweek as EW

# data[ew] is cumulative yearly value at epiweek ew
# Missing epiweek implies no recorded value at that week
# Returns dict mapping epiweek to weekly value, linearly smoothing missing counts
def cum_to_week(data):
    epiweeks = list(data.keys())
    all_epiweeks = list(EW.range_epiweeks(min(epiweeks),max(epiweeks),inclusive=True))
    
    result = np.zeros((len(all_epiweeks)))
    last_valid = (-1, 0) # (idx, value)
    for i in range(len(result)):
        ew = all_epiweeks[i]
        if ew in data:
            if data[all_epiweeks[i]] is not None:
                result[last_valid[0]+1:i+1] = (data[ew] - last_valid[1]) / float(i - last_valid[0]) # Evenly distribute missing counts
                last_valid = (i, data[ew])
        yr, wk = EW.split_epiweek(all_epiweeks[i])
        if EW.get_num_weeks(yr) == wk:
            result[last_valid[0]+1:i+1] = 0 # Fill rest of year with 0s, not getting this information
            last_valid = (i, 0) # Start new year at 0
    return {all_epiweeks[i]: result[i] for i in range(len(all_epiweeks))}

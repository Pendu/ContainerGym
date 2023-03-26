# JSON Configuration Files
By providing a .json file with environment parameters, we can create instances of the environment with the from_json function.  
Currently, there are two supported formats. One with named bunkers such as "C1-20", and one with "anonymous" bunkers, meaning that they are only ever referenced by their index and do not possess a name.

## Named Bunkers
In this format, all parameters that differ between bunkers must be provided in dictionaries.  
The keys of the dictionary must be the bunker name, and the values the corresponding parameters.  
The ENABLED_BUNKERS parameter must provide a list of the names of the enabled bunkers.

## Anonymous Bunkers
In this format, all parameters that differ between bunkers must be provided in lists.  
Each list index refers to the bunker at that index.  
The ENABLED_BUNKERS parameter must contain an integer value of the number of enabled bunkers. The provided lists must be at least of this length.

## Allowed Parameters  
`"MAX_EPISODE_LENGTH",
    "TIMESTEP",
    "ENABLED_BUNKERS",
    "N_PRESSES",
    "MIN_STARTING_VOLUME",
    "MAX_STARTING_VOLUME",
    "FAILURE_PENALTY",
    "RW_MUS",
    "RW_SIGMAS",
    "MAX_VOLUMES",
    "BALE_SIZES",
    "PRESS_OFFSETS",
    "PRESS_SLOPES",
    "REWARD_PARAMS",
    "MIN_REWARD"`
#importing the libraries

import numpy as np


#Setting the parameters for gamma and alpha for the Q-learning
gamma = 0.75
alpha = 0.9

#Part-1 DEFINING the environment

#defing the states

location_to_state = {"A": 0,
                     "B" : 1,
                     "C" : 2,
                     "D" : 3,
                     "E" : 4,
                     "F" : 5,
                     "G" : 6,
                     "H" : 7,
                     "I" : 8,
                     "J" : 9,
                     "K" : 10,
                     "L" : 11}

#defining actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

#defining the rewards
#it can be considered as a adjacency matrix with rewards.
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])





           
# AI-solution

#making a mapping from state to location

state_to_location = {state : location for location , state in location_to_state.items()}
#makng a function to return the optimal route

def route(starting_location,ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state,ending_state] = 1000
    Q = np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route
#going into production
    
def best_route(starting_location,intermediary_location,ending_location):
    return route(starting_location,intermediary_location) + route(intermediary_location,ending_location)[1:]
    

#printing the final route
print('Route:')
best_route('A' ,'G', 'D')    


        
    
    
    
    
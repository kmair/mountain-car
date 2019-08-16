from environment import MountainCar
import numpy as np
import sys

mode = sys.argv[1]
#weight_out = sys.argv[2]
#return_out = sys.argv[3]
#episodes = sys.argv[4]
#max_iter = sys.argv[5]
#epsilon = sys.argv[6]
#gamma = sys.argv[7]
#lr = sys.argv[8]

# =============================================================================
# 
# def main(args):
#     pass
# 
# if __name__ == "__main__":
#     main(sys.argv)
# =============================================================================

Car = MountainCar(mode)
SS = Car.state_space
AS = 3    # Action space = 3
w = np.zeros((SS,AS))
b = 0.
state = Car.reset()

LS = len(state)
States = state.keys()

episodes = 4
max_iter = 5
epsilon = 0.6
gamma = 0.7
lr = 0.1


def weight(mode, state, w):     # state is the dictionary here
    if mode == 'raw':
        V = []
        for i in range(SS):
            Val = w[i] * state[i]
            V.append(Val)
        SV = sum(V)

    if mode == 'tile':
        V = []
        for j in state.keys():
            Val = w[j] * state[j]
            V.append(Val)
        SV = sum(V)    

    return SV

Q_vals = weight(mode, state, w)
print(Q_vals)

def Action_select(q_vals, epsillon):
    a_Exploit = np.argmax(q_vals)
    a_Explore = np.random.randint(3, size=1)[0]
    A = np.array([a_Exploit, a_Explore])
    a = np.random.choice(A, 1, p=[1-epsillon, epsillon])
    return a
    
def Q_train(mode, state, w, b, alpha, gamma, epsillon):
    sTw = weight(mode, state, w)
    q_vals = sTw + b
    a = Action_select(q_vals, epsillon)
    Q = max(q_vals)
    Sprime, reward, done = MountainCar.step(Car, a)
    
    for noe in range(episodes):
        state = Car.reset()
        
        
        if done == 'True':
            break
            #state = Car.reset()
        
        elif done != 'True':
            pass
        
        '''Computing q_pi (s,a)'''
        Qprime = weight(mode, Sprime, w) + b
        Q_next = max(Qprime)
        
        '''Gradient'''
        dQ = np.zeros((SS, AS))
        s = np.zeros((SS))
        
        if mode == 'raw':
            for j in range(SS):
                s[j] = state[j]
                
        if mode == 'tile':
            for j in state.keys():
                s[j] = 1
        
        print(a)
        print(dQ[:, a])
        dQ[:, a] = np.array([s])
    
        '''UPDATE''' 
        grad = alpha * (Q - (reward + gamma*Q_next))
        w = w - grad * dQ
        b = b - alpha * (Q - (reward + gamma*Q_next)) * 1
    #print(state[0])
        msg = 1
        return w, b, Sprime, msg
    
W = Q_train(mode, state, w, b, 0.1, 0.1, 0.1)

#print(W)
def Q_learning(mode, state, w, b, alpha, gamma, epsillon, Ep):
    Num_episodes = 0
    w, b, State, msg = Q_train(mode, state, w, b, alpha, gamma, epsillon)
    if msg == 0:
        Num_episodes += 1
        
        return Num_episodes
    if Num_episodes <= Ep:
        pass
    
    w, b, State, msg = Q_learning(mode, State, w, b, alpha, gamma, epsillon, Ep)
    return w, b, State, msg 















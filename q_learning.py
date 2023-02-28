import numpy as np

class QLearning:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.9
        self.actions = np.array([0, 1, 2]) #0:HL, 1:HR, 2:STAY
        self.Q = np.zeros(5) #(状態,行動) = 0:(START,HL), 1:(START,HR), 2:(LL,STAY), 3:(LN,STAY), 4:(LR,STAY)
        self.V = np.zeros(4) #(状態) = 0:(START), 1:(LL), 2:(LN), 3(LR)
        self.td_error_v = np.zeros(4) #(状態) = 0:(START), 1:(LL), 2:(LN), 3(LR)

    def reset(self):
        self.Q = np.zeros(5)
        self.V = np.zeros(4)
        self.td_error_v = np.zeros(4)

    def act(self, state):
        self.state = state
        self.action = self.softmax()
        return self.action

    def update(self, state, action, reward, next_state):
        if action == 0 or action == 1: #action=0:HL or 1:HR, state=0:START
            #0:START, 1:LL, 2:LN, 3:LR
            max_Q = self.Q[next_state+1] #(状態,行動) = 2:(LL,STAY), 3:(LN,STAY), 4:(LR,STAY)
            td_error_q = reward + self.gamma * max_Q  - self.Q[action] #(状態,行動) = 0:(START,HL), 1:(START,HR)
            self.Q[action] += self.alpha * td_error_q
            td_error_v = reward + self.gamma * self.V[next_state] - self.V[state]
            self.V[state] += self.alpha * td_error_v
            self.td_error_v[state] = td_error_v
            return self.Q[0], self.Q[1], False, self.V[0], self.V[1], self.V[2], self.V[3], self.td_error_v
        else: #action=2:STAY
            td_error_v = reward - self.V[state]
            self.V[state] += self.alpha * td_error_v
            self.td_error_v[state] = td_error_v
            return self.Q[0], self.Q[1], True, self.V[0], self.V[1], self.V[2], self.V[3], self.td_error_v


    def softmax(self):
        if self.state == 0: #state=START
            CanChoiceQ = np.array(self.Q[:2]) #0:HL or 1:HR
            x = np.exp(CanChoiceQ)
            u = np.sum(x)
            p_softmax = x/u
            return np.random.choice([0, 1], p=p_softmax)
        else: #state=1:LL, 2:LN, 3:LR
            return 2 #action=2:LN
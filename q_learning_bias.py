import numpy as np
from q_learning import QLearning


class QLearningBias(QLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias_weight = kwargs['biasWeight']
        self.Q_bias = np.zeros(5)

    def reset(self):
        super().reset()
        self.Q_bias = np.zeros(5)

    def act(self, state):
        action = self.bias_softmax(state)
        return action

    def update(self, state, action, reward, next_state):
        if action == 0 or action == 1: #action=0:HL or 1:HR, state=0:START
            #0:START, 1:LL, 2:LN, 3:LR
            max_Q = self.Q[next_state+1] #(状態,行動) = 2:(LL,STAY), 3:(LN,STAY), 4:(LR,STAY)
            td_error_q = reward + self.gamma * max_Q - self.Q[action] #(状態,行動) = 0:(START,HL), 1:(START,HR)
            self.Q[action] += self.alpha * td_error_q
            self.td_error_q[action] = td_error_q
            td_error_v = reward + self.gamma * self.V[next_state] - self.V[state]
            self.V[state] += self.alpha * td_error_v
            self.td_error_v[state] = td_error_v
            self.Q_bias[action] = self.Q[action] + np.abs(td_error_q) * self.bias_weight ###Q値に何かしらのバイアスを掛ける
            return self.Q, self.V, self.td_error_q, self.td_error_v
        else: #action=2:STAY, next_state=4:TN, state=1:(LL), 2:(LN), 3(LR)
            td_error_q = reward - self.Q[state+1]
            self.Q[state+1] += self.alpha * td_error_q #(状態,行動) = 2:(LL,STAY), 3:(LN,STAY), 4:(LR,STAY)
            self.td_error_q[state+1] = td_error_q
            td_error_v = reward - self.V[state]
            self.V[state] += self.alpha * td_error_v
            self.td_error_v[state] = td_error_v
            self.Q_bias[state+1] = self.Q[state+1] + np.abs(td_error_q) * self.bias_weight ###Q値に何かしらのバイアスを掛ける
            return self.Q, self.V, self.td_error_q, self.td_error_v

    def bias_softmax(self, state):
        if state == 0: #state=START
            CanChoiceQ = np.array(self.Q_bias[:2]) #0:HL or 1:HR
            CanChoiceQ -= np.max(CanChoiceQ)
            exp_CanChoiceQ = np.exp(CanChoiceQ)
            prob = exp_CanChoiceQ / np.sum(exp_CanChoiceQ)
            return np.random.choice([0, 1], p=prob)
        else: #state=1:LL, 2:LN, 3:LR
            return 2 #action=2:STAY
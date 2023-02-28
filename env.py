import numpy as np

class Environment:
    def __init__(self):
        self.map = np.array([0, 1, 2, 3]) #0:START, 1:LL, 2:LN, 3:LR
        self.start_position = self.map[0] #START=0
        self.agent_position = self.start_position

    def reset(self):
        self.agent_position = self.start_position
        return self.agent_position, False

    def update_state(self, action):
        if action == 0: #action=HL, p=0.5でLL遷移, p=0.5でLN遷移
            if np.random.rand() >= 0.5:
                self.agent_position = 1 #LLに遷移
                return self.agent_position
            else:
                self.agent_position = 2 #LNに遷移
                return self.agent_position
        elif action == 1: #action=HR, p=1でLRに遷移
            self.agent_position = 3
            return self.agent_position

    def step(self, state, action):
        next_state = self.update_state(action)
        reward = self.reward()
        return reward, next_state

    def reward(self):
        if self.agent_position == 1: #状態がLL
            return 1
        elif self.agent_position == 2: #状態がLN
            return 0
        elif self.agent_position == 3: #状態がLR
            if np.random.rand() >= 0.5: #p=0.5で報酬1or0
                return 1
            else:
                return 0
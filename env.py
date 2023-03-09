import numpy as np


class Environment:
    def __init__(self):
        self.map = np.array([0, 1, 2, 3, 4]) #0:START, 1:LL, 2:LN, 3:LR, 4:TN
        self.start_position = self.map[0] #START=0
        self.agent_position = self.start_position
        self.past_agent_position = self.agent_position
        self.past_action = None

    def reset(self):
        self.agent_position = self.start_position
        self.past_agent_position = self.agent_position
        self.past_action = None
        return self.agent_position, False

    def update_state(self, action):
        if action == 0: #action=HL, p=0.5でLL遷移, p=0.5でLN遷移
            self.past_action = action
            if np.random.rand() >= 0.5:
                self.past_agent_position = self.agent_position
                self.agent_position = 1 #LLに遷移
            else:
                self.past_agent_position = self.agent_position
                self.agent_position = 2 #LNに遷移
        elif action == 1: #action=HR, p=1でLRに遷移
            self.past_action = action
            self.past_agent_position = self.agent_position
            self.agent_position = 3
        else:
            self.past_agent_position = self.agent_position
            self.agent_position = 4 #action=STAY, TNに遷移

    def step(self, action):
        self.update_state(action)
        return self.agent_position, self.reward(), (self.agent_position == 4)

    def reward(self):
        if self.agent_position == 4:
            if self.past_action == 0:
                if self.past_agent_position == 1:
                    return 1
                elif self.past_agent_position == 2:
                    return 0
            elif self.past_action == 1:
                if np.random.rand() >= 0.5: #p=0.5で報酬1or0
                    return 1
                else:
                    return 0
        else:
            return 0
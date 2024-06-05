import numpy as np
import random

# Define the grid world environment
class GridWorldEnv:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)  # Starting position
        self.goal = (size - 1, size - 1)  # Goal position
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        if action == 0:  # Up
            self.state = (max(0, self.state[0] - 1), self.state[1])
        elif action == 1:  # Down
            self.state = (min(self.size - 1, self.state[0] + 1), self.state[1])
        elif action == 2:  # Left
            self.state = (self.state[0], max(0, self.state[1] - 1))
        elif action == 3:  # Right
            self.state = (self.state[0], min(self.size - 1, self.state[1] + 1))
        
        if self.state == self.goal:
            reward = 1
            done = True
        else:
            reward = -0.01  # Small penalty to encourage faster completion
            done = False
        
        return self.state, reward, done
    
    def render(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        grid[self.state] = 1
        grid[self.goal] = 9
        print(grid)

# Define the hierarchical agent
class HierarchicalAgent:
    def __init__(self, env):
        self.env = env
        self.high_level_policy = self.create_high_level_policy()
        self.low_level_policy = self.create_low_level_policy()
    
    def create_high_level_policy(self):
        # A simple policy to set sub-goals
        policy = {}
        for i in range(self.env.size):
            for j in range(self.env.size):
                if i < self.env.size - 1:
                    policy[(i, j)] = (i + 1, j)
                else:
                    policy[(i, j)] = (i, j + 1) if j < self.env.size - 1 else (i, j)
        return policy
    
    def create_low_level_policy(self):
        # A simple policy to move towards the sub-goal
        return lambda s, g: random.choice(self.get_possible_actions(s, g))
    
    def get_possible_actions(self, state, goal):
        actions = []
        if state[0] < goal[0]:
            actions.append(1)  # Move down
        if state[0] > goal[0]:
            actions.append(0)  # Move up
        if state[1] < goal[1]:
            actions.append(3)  # Move right
        if state[1] > goal[1]:
            actions.append(2)  # Move left
        return actions
    
    def select_subgoal(self, state):
        return self.high_level_policy[state]
    
    def select_action(self, state, subgoal):
        return self.low_level_policy(state, subgoal)
    
    def train(self, episodes=100):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                subgoal = self.select_subgoal(state)
                action = self.select_action(state, subgoal)
                next_state, reward, done = self.env.step(action)
                state = next_state
                self.env.render()
                if done:
                    print(f"Episode {episode + 1} finished")
                    break

# Instantiate the environment and the agent
env = GridWorldEnv()
agent = HierarchicalAgent(env)

# Train the hierarchical agent
agent.train()

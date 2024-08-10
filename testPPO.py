from ppoagent import Network
import torch
from preprocess import preProcess
import argparse
import gym
from collections import deque
import numpy as np

EPISODES = 1000
SKIP_INTIAL_FRAMES = 50
FRAMES_PER_ACTION = 25

model = Network((1,96,96),3)
print(model)
model.load_state_dict(torch.load(r"C:\Users\Dell\Downloads\ppo_actor.pth"))
env = gym.make("CarRacing-v2",render_mode="human")
for episode in range(EPISODES):

    init_state = env.reset()[0]

    

    total_reward = 0
    negative_reward_counter = 0
    time_step = 1

    current_state = init_state
    current_state = preProcess.process(current_state)

    for i in range(SKIP_INTIAL_FRAMES):
        env.step([0,0,0])

    while True:
        env.render()
        current_state=np.float32(current_state)
        current_state = torch.tensor(current_state.reshape((1,96,96)))
        
        if len(current_state.shape)<4:
            current_state=torch.unsqueeze(current_state, 0)

        action = model(current_state).detach().numpy()[0]
        print(action)
        reward = 0
        for i in range(FRAMES_PER_ACTION):
            next_state, r, done, info,d = env.step(action)
            reward +=r
        
        
        next_state = preProcess.process(next_state)
        current_state = next_state
        
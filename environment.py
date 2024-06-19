import argparse
import gym
from collections import deque
from agent import agent,memory
from preprocess import preProcess
import numpy as np

EPISODES = 1000
SKIP_INTIAL_FRAMES = 50
FRAMES_PER_ACTION = 25
BATCH_SIZE = memory.size

#REMEMBER TO CHANGE ACTION SPACE

env = gym.make("CarRacing-v2",render_mode="human")
agent.model.summary()
agent.model.load_weights(r"C:\Users\Dell\Downloads\DQN700.h5")
print("S0a")
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


        action = agent.act(current_state)
        
        reward = 0
        for i in range(FRAMES_PER_ACTION):
            next_state, r, done, info,d = env.step(action)
            reward +=r
        
        total_reward += reward
        print(reward)
        if reward<0:
            negative_reward_counter+=1
        else:
            negative_reward_counter = 0
        
        if total_reward<0:
            print("TOTAL_REWARD")
            break
        if negative_reward_counter>5:
            print("NEG REWARD",negative_reward_counter)
            break
        next_state = preProcess.process(next_state)
        memory.add([current_state,action,reward,next_state])
        current_state = next_state
        
        if memory.size>=BATCH_SIZE:
            print("LEARNING",total_reward)
            agent.learn()
            agent.update_model()
    print(episode)
    if episode%20==0:
        print("Saved")
        agent.save_model("DQN",episode)
import argparse
import gym
from collections import deque
from agent import agent,memory
from preprocess import preProcess


EPISODES = 25
SKIP_INTIAL_FRAMES = 50

env = gym.make("CarRacing-v2",render_mode="human")
agent.model.summary()
agent.model.load_weights(r"C:\Users\Dell\Downloads\77.h5")

print("Sa")
for e in range(EPISODES):

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
        
        next_state, reward, done, info,d = env.step(action)

        total_reward += reward
        next_state = preProcess.process(next_state)


        memory.add([current_state,action,reward,next_state])


        current_state = next_state
        if time_step%5==0:
            agent.update_model()

        if done:
            break
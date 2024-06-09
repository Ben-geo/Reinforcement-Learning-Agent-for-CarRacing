
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import cv2
from preprocess import preProcess

BATCH_SIZE = 64
class Memory():
    def __init__(self, max_size):
        self.size = max_size
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = buffer_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]
memory=Memory(BATCH_SIZE)

class Agent:
    def __init__(
        self,
        action_space    = [
                    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
                    (-1, 1,   0), (0, 1,   0), (1, 1,   0), #           (Steering, Gas, Break)
                    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range       -1~1     0~`1`   0~1
                    (-1, 0,   0), (0, 0,   0), (1, 0,   0)
                    ],
        
    ):
        action_space = []
        for steering in [0,0.33,0.67,1]:
            for accel in [0,0.33,0.67,1]:
                for braking in [0,0.33,0.67,1]:
                    action_space.append((steering,accel,braking))
                    action_space.append((-steering,accel,braking))
        self.action_space    = action_space

        self.epsilon = 0.2
        self.gamma = 0.95
        self.learning_rate = 0.01

        self.model           = self.build_model()
        self.target_model    = tf.keras.models.clone_model( self.model )


    def build_model( self ):
        """Sequential Neural Net with x2 Conv layers, x2 Dense layers using RELU and Huber Loss"""
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=3, activation='relu', input_shape=(96, 96, 1)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3),strides=3, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(3, 3),strides=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model
    
    def act(self,state):
        if np.random.rand()>self.epsilon:
            allQ = self.model.predict(tf.expand_dims(state,0),verbose = 0)
            Q = np.argmax(allQ)
        else:
            Q = random.randrange(len(self.action_space))
        return self.action_space[Q]
    
    def update_model( self ):
        self.target_model.set_weights( self.model.get_weights() )

    def learn(self):
        train_state = []
        train_reward = []
        for state,action,reward,next_state in memory.sample(min(memory.size,64)):
            
            target = agent.model.predict(tf.expand_dims(state,0),verbose = 0)[0]
            
            
            t = self.target_model.predict(np.expand_dims(next_state, axis=0),verbose = 0)[0]
            target[ self.action_space.index(action) ] = reward + self.gamma * np.amax(t)

            train_state.append(state)
            train_reward.append(target)

        train_state=np.array(train_state)
        train_reward=np.array(train_reward)
        self.model.fit(train_state,train_reward,epochs=1,verbose = 0)
        
agent=Agent()
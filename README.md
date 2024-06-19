# Proximal Policy Optimization and Deep Q Learning in OpenAi Environments
**Business Objective :** <br>
 To implement and evaluate Proximal Policy Optimization and Deep Q learning accross various Open AI environments

**File Descriptions :** <br>
* **agent.py :** Deep Q agent
* **environment.py :**  Environment to test the models on
* **ppoagent.ipynb :** Proximal Policy Optimzation Agent
* **preprocess.py :** Functions to preprocess the states


### Deep Q Learning


![](/utils/images/DeepQ.jpg)
Deep Q-Learning is a type of reinforcement learning algorithm that uses a deep neural network to approximate the Q-function, which is used to determine the optimal action to take in a given state. The Q-function represents the expected cumulative reward of taking a certain action in a certain state and following a certain policy. In Q-Learning, the Q-function is updated iteratively as the agent interacts with the environment. Deep Q-Learning is used in various applications such as game playing, robotics and autonomous vehicles.

Experience replay is a technique where the agent stores a subset of its experiences (state, action, reward, next state) in a memory buffer and samples from this buffer to update the Q-function. This helps to decorrelate the data and make the learning process more stable. Target networks, on the other hand, are used to stabilize the Q-function updates. In this technique, a separate network is used to compute the target Q-values, which are then used to update the Q-function network.


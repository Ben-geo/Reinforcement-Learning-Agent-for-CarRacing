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

### Proximal Policy Optimization

Main changes : 
* **Policy-Based Method** - 
    Policy : a function that maps states to action
        PPO focuses on this funcion itself
* **Clipped Objective** -
    revent the policy from changing too much in a single update.

#### Steps 
* Initial Policy: At the start, the robot's policy is random. It might take some steps correctly but often fall.
* Collecting Experience: The robot interacts with the environment (walks, falls, walks again), and PPO collects this experience—tracking the states, actions, and rewards.
Updating the Policy: PPO then updates the policy based on this experience. The update is done in a way that improves the policy, but not too drastically, to ensure the robot's learning is stable.
    * Clipping: PPO introduces a clipping mechanism to ensure that the robot's policy doesn't change too much in one step, avoiding situations where the robot suddenly starts behaving erratically.
* Improved Policy: Over time, the policy gets better. The robot starts to fall less often and walks more steadily.
![](/utils/images/DeepQ.jpg)
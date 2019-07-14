"""
Policy Gradient, Reinforcement Learning.
The cart pole example
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    output_graph=True,
)

# you can change the total number of episode, but after 82 episode it is already good enough



i_episode =0


observation = env.reset()
layer1_weight = RL.get_w1(observation)
layer2_weight = RL.get_w2(observation)
layer1_bias = RL.get_b1(observation)
layer2_bias = RL.get_b2(observation)

layer1_output = RL.first_layer_output(observation)
layer2_output = RL.second_layer_output(observation)

softmax_out = RL.softmax_output(observation)


print("episode:", i_episode, "  observation_:",   observation)
print ("the parameters before training: ")
print ("the layer1_ weights is :", layer1_weight)
print ("the layer2_ weights is :", layer2_weight)

print ("the layer1_ bias is :", layer1_bias)
print ("the layer2_ bias is :", layer2_bias)

print ("the layer1_output  is :", layer1_output)
print ("the layer2_output  is :", layer2_output)
print ("the softmax output  is :", softmax_out)

step = 0

#  here only play one episode
if True:
        # if RENDER: env.render()
    while True:    
        action = RL.choose_action(observation)


        ## here only for debug to see the forward propagation and the weights
        #layer1_output = RL.first_layer_output(observation)
        #layer1_weight = RL.get_w1(observation)
        #print ("the layer1 output  is :", layer1_output)
        #print ("the layer1 weight now is :", layer1_weight)


        observation_, reward, done, info = env.step(action)
        print("episode:", i_episode, "  reward:", reward)
        print("episode:", i_episode, "  done:",   done)
        print("episode:", i_episode, "  observation_:",   observation_)
        #when it runs, the reward is always 1
        #when it falls or failure or you call one episode is end, the reward is 0
        softmax_out = RL.softmax_output(observation)
        print (" in the ", step, "step:")
        print ("the softmax output  is :", softmax_out)

        step = step +1


        RL.store_transition(observation, action, reward)
        
        # print("action space is:", RL.n_actions) is 2
        
        
        
        if done:
        #if step == 1:
            print ("")
            print ("############ the", i_episode, "episode is finished now, caculate the rewards for this episode")
            ep_rs_sum = sum(RL.ep_rs)
            
            
            
            if 'running_reward' not in globals():
                print(" running reward is not in globals")
                running_reward = ep_rs_sum
                print(" running_reward = ep_rs_sum is : ", running_reward)
            # only the first time defines here, then it goes to the else case every time    
            else:
                print(" running reward is in globals")
                print(" the current running_reward is : ", running_reward)
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                print(" running_reward = running_reward * 0.99 + ep_rs_sum * 0.01 : ", running_reward)
            
            
            
            # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            # if you want to see the car pole running , you can uncommit the above sentence
            
            print("episode:", i_episode, "  currint reward:", int(running_reward))
            print("episode:", i_episode, "  ep_rs_sum:", int(ep_rs_sum))

            # at this position, we already have all the information that can calculate the backpropogation:
            # the first reward is [5.5203495]
            # we can first just use this to calculate the weight update



            
            print ("############ now start the learning process")
            vt = RL.learn(observation)
            
            
            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        
        
        
        observation = observation_
                # here is very important to keep the code alignment
                # otherwise the agent is not learning, and the keeping pole will finished at around 9 or 10 steps
        
        
        
print("########### the end of the first episode #############3")


layer1_weight = RL.get_w1(observation)
layer2_weight = RL.get_w2(observation)

layer1_bias = RL.get_b1(observation)
layer2_bias = RL.get_b2(observation)

layer1_output = RL.first_layer_output(observation)
layer2_output = RL.second_layer_output(observation)

softmax_out = RL.softmax_output(observation)

print ("the end of one training : ")
print ("the layer1_ weights is :", layer1_weight)
print ("the layer2_ weights is :", layer2_weight)

print ("the layer1_ bias is :", layer1_bias)
print ("the layer2_ bias is :", layer2_bias)

print ("the layer1_output  is :", layer1_output)
print ("the layer2_output  is :", layer2_output)
print ("the softmax output  is :", softmax_out)

        
        
        
        
## try the manually training process:

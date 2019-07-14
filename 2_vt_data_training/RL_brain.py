"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
But change this code only for softmax layer training.
Policy Gradient, Reinforcement Learning.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        
        # fc1
        self.layer1_output = tf.layers.dense(
            inputs=self.tf_obs,
            units=4,
            # the structure changed here
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        self.layer2_output = tf.layers.dense(
            inputs=self.layer1_output,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )



        self.all_act_prob = tf.nn.softmax(self.layer2_output , name='act_prob')  # use softmax to convert to probability
        # [[-0.66178095 -0.72552913]
        # [-0.551769   -0.85785687]
        
        # extract the first layer weights here
        with tf.variable_scope('fc1', reuse=True):
            self.w1 = tf.get_variable('kernel')
        # extract the second layer weights here
        with tf.variable_scope('fc2', reuse=True):
            self.w2 = tf.get_variable('kernel')
        
        with tf.variable_scope('fc1', reuse=True):
            self.b1 = tf.get_variable('bias')
        # extract the second layer weights here
        with tf.variable_scope('fc2', reuse=True):
            self.b2 = tf.get_variable('bias')



        

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   
            # this is negative log of chosen action
            
            # or in this way:
            # this way is the recommended way and more clear
            neg_log_prob = tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # neg_log_prob = tf.reduce_sum [0.72552913 0.85785687]
            
            # added here for debug
            # with tf.Session() as sess:
            #    print sess.run(x)

            # loss function defines here
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            


            print ("the loss is :", loss)
            '''                   
               [ 3.4827876   3.4188704   2.360828    1.8449908   1.2909881   0.6933805
               0.04737188 -0.6509731  -1.4045806  -1.6746713  -2.94834    -2.7844634
               -4.592379  ]
                the total loss is tf.reduce_mean(neg_log_prob * self.tf_vt) 
               -0.07047617, is the average of the neg_log_prob * vt
            '''


            ################ data debug  #####################
            # multiply test for debug, not influence the main process
            multiply = neg_log_prob * self.tf_vt
            print ("the shape of the multiply is", np.shape(multiply))



        #### change the optimizer here ####
        with tf.name_scope('train'):
          #self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
          self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
          # when the learning rate is 0.01, the learning process is  very long
          # when change the learning rate to 0.1, the process is much faster, but still slower than AdamOptimizer











    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def forward_cpu(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # return the output of the neural network
        # to be specific, the output of the softmax
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return prob_weights

    def first_layer_output(self, observation):
        layer1 = self.sess.run(self.layer1_output, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # return the output of the neural network
        # to be specific, the output of the softmax
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return layer1


    def second_layer_output(self, observation):
        layer2 = self.sess.run(self.layer2_output, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # return the output of the neural network
        # to be specific, the output of the softmax
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return layer2


    def softmax_output(self, observation):
        softmax_out = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        return softmax_out

    def get_w1(self, observation):
        weights1 = self.sess.run(self.w1, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # return the output of the neural network
        # to be specific, the output of the softmax
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return weights1

    def get_w2(self, observation):
        weights2 = self.sess.run(self.w2, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # return the output of the neural network
        # to be specific, the output of the softmax
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return weights2 



    def get_b1(self, observation):
        bias1 = self.sess.run(self.b1, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # return the output of the neural network
        # to be specific, the output of the softmax
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return bias1

    def get_b2(self, observation):
        bias2 = self.sess.run(self.b2, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # return the output of the neural network
        # to be specific, the output of the softmax
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return bias2 

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self, observation):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        print ("discounted_ep_rs_norm is return by the self built-in function:", discounted_ep_rs_norm)
        # print ("now in the learn function, the self.ep_obs is", self.ep_obs )
        # print ""
        print ("the shape of the nparray.ep_obs is", np.shape(self.ep_obs))

        '''
        # here the optimizer is AdamOptimizer
        # with tf.name_scope('train'):
        #     self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        # train on one complete episode
        self.sess.run(self.train_op, feed_dict={    # train_op is the ASdam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        # here the tf_vt is feed by the normalied reward at this episode
        '''
                #################  the debugging signals ################################
        print ("np.shape(self.ep_as)", np.shape(self.ep_as))
        print ("np.shape(self.tf_acts)", np.shape(self.tf_acts))
        print ("np.shape(self.tf_vt)", np.shape(self.tf_vt))
        # print ("np.array(self.ep_as)", np.array(self.ep_as))
        # [0 1 1 0 1 0 1] , shows the actions that have taken

        ######### uncommit below,  here below for debug
        
        ###################  the six inportant signals ##########################
        tensor_tf_acts = self.tf_acts
        # [0 1 0 0 0 0 0 0 0 1 0 1 0]
        one_hot_action = tf.one_hot(self.tf_acts, self.n_actions)


        action_probability = self.all_act_prob
        every_step_reward = self.tf_vt
        """
        print("")
        print("all the parameters before we run sess run train.op")

        print ("self.tf_obs = np.vstack(self.ep_obs, the input is :")
        print(self.sess.run(self.tf_obs, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        print ("self.tf_acts = the action takes :")
        print(self.sess.run(self.tf_acts, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        print ("self.tf_vt = the reword gets :")
        print(self.sess.run(self.tf_vt, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))
        """

        print ("tensor_tf_acts = self.tf_acts = np.array(self.ep_as) is :")
        print(self.sess.run(tensor_tf_acts, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))
        # [0 1 1 1 0 0 1 1 0 0 1 0 1 0 0 0 0 0 1]
        # self.tf_acts stores all the actions that have taken



        print ("one_hot_action = tf.one_hot(self.tf_acts, self.n_actions):")
        print(self.sess.run(one_hot_action, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        print ("action_probability = tf.all_act_prob: ")
        print(self.sess.run(action_probability, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        print ("every_step_reward = tf.vt: ")
        print(self.sess.run(every_step_reward, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        log_all_act_prob = tf.log(self.all_act_prob)
        print ("log_all_act_prob = log (tf.all_act_prob) : ")
        print(self.sess.run(log_all_act_prob, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        prob_mult_one_hot = tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions)
        print ("prob_mult_one_hot = tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions) : ")
        print(self.sess.run(prob_mult_one_hot, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        array_input = self.tf_obs
        print ("array_input , tf.obs = the full stack observation is: ")
        print(self.sess.run(array_input , feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))



        """
        reduce_sum_debug = tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        print ("reduce_sum_debug = neg_log_prob = tf.reduce_sum : ")
        print(self.sess.run(reduce_sum_debug, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        neg_log_prob = tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions), axis=1)

        print ("the neg_log_prob is tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions )")
        print(self.sess.run(neg_log_prob, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        multiply_ = neg_log_prob * self.tf_vt
        print ("the multiply_ multiply_ = neg_log_prob * self.tf_vt ")
        print(self.sess.run(multiply_, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))




        print ("the total loss is tf.reduce_mean(neg_log_prob * self.tf_vt) ")
        print(self.sess.run(tf.reduce_mean(neg_log_prob * self.tf_vt), feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        """
        ############ ******************* the first action and step training happens here ***********  ############

        # action_probability = self.all_act_prob
        Y1 = action_probability[0][0]  # the first unit output 
        Y2 = action_probability[0][1]  # the second unit output

        print ("the action prob is Y1:", action_probability[0][0])
        print(self.sess.run(Y1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the action prob is Y2:", action_probability[0][1])
        print(self.sess.run(Y2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        
        x1 = array_input[0][0]
        x2 = array_input[0][1]
        x3 = array_input[0][2]
        x4 = array_input[0][3]
        print ("the input 1 is x1:", x1)
        print(self.sess.run(x1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print ("the input 2 is x2:", x2)
        print(self.sess.run(x2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print ("the input 3 is x3:", x3)
        print(self.sess.run(x3, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print ("the input 4 is x4:", x4)
        print(self.sess.run(x4, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

  
        print(" the w2 is the first layer weight: ", self.w2)
        print(self.sess.run(self.w2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        weight2 = self.w2
        print ("the weight2[0][0] is:", weight2[0][0])
        print(self.sess.run(weight2[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        weight1 = self.w1
        print ("the weight1 (whole 1st layer) is:", weight1)
        print(self.sess.run(weight1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))



        tanh_delta0 = (1-self.layer1_output[0][0]*self.layer1_output[0][0])
        tanh_delta1 = (1-self.layer1_output[0][1]*self.layer1_output[0][1])
        tanh_delta2 = (1-self.layer1_output[0][2]*self.layer1_output[0][2])
        tanh_delta3 = (1-self.layer1_output[0][3]*self.layer1_output[0][3])


        print ("the layer1_output (whole) is :", self.layer1_output)
        print(self.sess.run(self.layer1_output, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_output[0][0] is :", self.layer1_output[0][0])
        print(self.sess.run(self.layer1_output[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the tanh_delta is :", tanh_delta0)
        print(self.sess.run(tanh_delta0, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        ############# here start to calculate the diff layer by layer #######################
        diff_start1 = -every_step_reward[0]*one_hot_action[0][0]   # 5.5203
        diff_start2 = -every_step_reward[0]*one_hot_action[0][1]   # 0
        print ( " diff-start is :")

        
        # one_hot_action[0][0]
        # one_hot_action[0][0] one_hot_action[0][1] one_hot_action[1][0] one_hot_action[1][1]   is 1  0  0  1
        
        print(self.sess.run(diff_start1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print(self.sess.run(diff_start2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        """
        print(self.sess.run(one_hot_action[1][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print(self.sess.run(one_hot_action[1][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        """



        #tf_const_diff_soft = tf.constant([[0.0,0.0,0.0],[0.0,0.0,0.0,],[0.0,0.0,0.0]])

        #diff_soft = tf.get_variable("diff_soft", dtype=tf.float32,initializer=tf_const_diff_soft)
        diff_soft = [[0.0, 0.0], [0.0, 0.0]]
        # 13 is the number of steps in one episode
        diff_soft11 = (1-Y1)/13
        diff_soft12 = -Y2/13
        diff_soft21 = -Y1/13
        diff_soft22 = (1-Y2)/13
        diff_soft[0][0] = diff_soft11 * diff_start1
        diff_soft[0][1] = diff_soft12 * diff_start1
        diff_soft[1][0] = diff_soft21 * diff_start2
        diff_soft[1][1] = diff_soft22 * diff_start2





        
        print (" ")
        print ("*******************")
        print ("the diff_soft is:")
        print(self.sess.run(diff_soft, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        


        
        delta_layer2_w11 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][0]
        delta_layer2_w12 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][0]


        delta_layer2_w21 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][1]
        delta_layer2_w22 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][1]

        delta_layer2_w31 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][2]
        delta_layer2_w32 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][2]


        delta_layer2_w41 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][3]
        delta_layer2_w42 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][3]


        layer2_w = [[0.0, 0.0], [0.0, 0.0],[0.0, 0.0],[0.0, 0.0]]
        layer1_w = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0,0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]]


        # layer2_w[0][1]
        print ("the weight2[0][1] #### before was :")
        
        print(self.sess.run(weight2[0][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the weight1[0][0] #### before was :")        
        print(self.sess.run(weight1[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        # here this delta weights also needs to declear after we know how many steps take in this episode

        delta_layer1_w11 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w12 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w13 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w14 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]

        delta_layer1_w21 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w22 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w23 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w24 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]

        delta_layer1_w31 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w32 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w33 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w34 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]

        delta_layer1_w41 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w42 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w43 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]
        delta_layer1_w44 = [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0]

        delta_layer1_total_w11 = 0.0
        delta_layer1_total_w12 = 0.0
        delta_layer1_total_w13 = 0.0
        delta_layer1_total_w14 = 0.0

        delta_layer1_total_w21 = 0.0
        delta_layer1_total_w22 = 0.0
        delta_layer1_total_w23 = 0.0
        delta_layer1_total_w24 = 0.0

        delta_layer1_total_w31 = 0.0
        delta_layer1_total_w32 = 0.0
        delta_layer1_total_w33 = 0.0
        delta_layer1_total_w34 = 0.0

        delta_layer1_total_w41 = 0.0
        delta_layer1_total_w42 = 0.0
        delta_layer1_total_w43 = 0.0
        delta_layer1_total_w44 = 0.0



        '''
        layer2_w[0][0] = weight2[0][0] - 0.1 * delta_layer2_w11
        layer2_w[0][1] = weight2[0][1] - 0.1 * delta_layer2_w12
        layer2_w[1][0] = weight2[1][0] - 0.1 * delta_layer2_w21
        layer2_w[1][1] = weight2[1][1] - 0.1 * delta_layer2_w22
        layer2_w[2][0] = weight2[2][0] - 0.1 * delta_layer2_w31
        layer2_w[2][1] = weight2[2][1] - 0.1 * delta_layer2_w32
        layer2_w[3][0] = weight2[3][0] - 0.1 * delta_layer2_w41
        layer2_w[3][1] = weight2[3][1] - 0.1 * delta_layer2_w42
        '''

        
        # please see the note book, the first example for w11 in the first layer
        # delta_layer1_w11[0] here in the implementation, the [0] means the first step, 
        # so after we can see the delta_layer1_w11[11], delta_layer1_w11[12], the 13th step
        delta_layer1_w11[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x1
        delta_layer1_w12[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x1
        delta_layer1_w13[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x1
        delta_layer1_w14[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x1



        delta_layer1_w21[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x2
        #  actually the w21 is more close to the w11, we should keep them together     

        delta_layer1_w22[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x2
        delta_layer1_w23[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x2
        delta_layer1_w24[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x2



        delta_layer1_w31[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x3
        delta_layer1_w32[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x3
        delta_layer1_w33[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x3
        delta_layer1_w34[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x3


        delta_layer1_w41[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x4
        delta_layer1_w42[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x4
        delta_layer1_w43[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x4
        delta_layer1_w44[0] = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x4










        # keep the update in the last step, here we can not update the weights
        print ("***********************in the first step ******************")

        
        step_num = 1
        # here set to one only for the for loop test


        # the range for loop from the 0 to step_num-1 
        for num in range(0,step_num):
            print ("num is", num)

        for num in range(0,step_num):
            delta_layer1_total_w11 = delta_layer1_total_w11 + delta_layer1_w11[num] 

        for num in range(0,step_num):
            delta_layer1_total_w12 = delta_layer1_total_w12 + delta_layer1_w12[num]
    
        for num in range(0,step_num):
            delta_layer1_total_w13 = delta_layer1_total_w13 + delta_layer1_w13[num]

        for num in range(0,step_num):
            delta_layer1_total_w14 = delta_layer1_total_w14 + delta_layer1_w14[num]

        print ("here we are in the first step")
        print ("the delta_layer1_total_w11 of the 1st layer w11 is :")
        print(self.sess.run(delta_layer1_total_w11, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("here we are in the first step")
        print ("the delta_layer1_w11[0] of the 1st layer w11 is :")
        print(self.sess.run(delta_layer1_w11[0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))



        print ("the delta_layer1_total_w12 of the 1st layer w12 is :")
        print(self.sess.run(delta_layer1_total_w12, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the delta_layer1_total_w13 of the 1st layer w13 is :")
        print(self.sess.run(delta_layer1_total_w13, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the delta_layer1_total_w14 of the 1st layer w14 is :")
        print(self.sess.run(delta_layer1_total_w14, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))



        for num in range(0,step_num):
            delta_layer1_total_w21 = delta_layer1_total_w21 + delta_layer1_w21[num] 

        for num in range(0,step_num):
            delta_layer1_total_w22 = delta_layer1_total_w22 + delta_layer1_w22[num]
    
        for num in range(0,step_num):
            delta_layer1_total_w23 = delta_layer1_total_w23 + delta_layer1_w23[num]

        for num in range(0,step_num):
            delta_layer1_total_w24 = delta_layer1_total_w24 + delta_layer1_w24[num]




        for num in range(0,step_num):
            delta_layer1_total_w31 = delta_layer1_total_w31 + delta_layer1_w31[num] 

        for num in range(0,step_num):
            delta_layer1_total_w32 = delta_layer1_total_w32 + delta_layer1_w32[num]
    
        for num in range(0,step_num):
            delta_layer1_total_w33 = delta_layer1_total_w33 + delta_layer1_w33[num]

        for num in range(0,step_num):
            delta_layer1_total_w34 = delta_layer1_total_w34 + delta_layer1_w34[num]




        for num in range(0,step_num):
            delta_layer1_total_w41 = delta_layer1_total_w41 + delta_layer1_w41[num] 

        for num in range(0,step_num):
            delta_layer1_total_w42 = delta_layer1_total_w42 + delta_layer1_w42[num]
    
        for num in range(0,step_num):
            delta_layer1_total_w43 = delta_layer1_total_w43 + delta_layer1_w43[num]

        for num in range(0,step_num):
            delta_layer1_total_w44 = delta_layer1_total_w44 + delta_layer1_w44[num]


        print ("the weight1[0][0] of the 1st layer w11 is :")
        print(self.sess.run(weight1[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        


        layer1_w[0][0] = weight1[0][0] - 0.1 * delta_layer1_total_w11
        layer1_w[0][1] = weight1[0][1] - 0.1 * delta_layer1_total_w12
        layer1_w[0][2] = weight1[0][2] - 0.1 * delta_layer1_total_w13
        layer1_w[0][3] = weight1[0][3] - 0.1 * delta_layer1_total_w14

        layer1_w[1][0] = weight1[1][0] - 0.1 * delta_layer1_total_w21
        layer1_w[1][1] = weight1[1][1] - 0.1 * delta_layer1_total_w22
        layer1_w[1][2] = weight1[1][2] - 0.1 * delta_layer1_total_w23
        layer1_w[1][3] = weight1[1][3] - 0.1 * delta_layer1_total_w24

        layer1_w[2][0] = weight1[2][0] - 0.1 * delta_layer1_total_w31
        layer1_w[2][1] = weight1[2][1] - 0.1 * delta_layer1_total_w32
        layer1_w[2][2] = weight1[2][2] - 0.1 * delta_layer1_total_w33
        layer1_w[2][3] = weight1[2][3] - 0.1 * delta_layer1_total_w34


        layer1_w[3][0] = weight1[3][0] - 0.1 * delta_layer1_total_w41
        layer1_w[3][1] = weight1[3][1] - 0.1 * delta_layer1_total_w42
        layer1_w[3][2] = weight1[3][2] - 0.1 * delta_layer1_total_w43
        layer1_w[3][3] = weight1[3][3] - 0.1 * delta_layer1_total_w44
        


        
        print ("the layer1_w[0][0] of the 1st layer w11 is :")
        print(self.sess.run(layer1_w[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[0][1] of the 1st layer w12 is :")
        print(self.sess.run(layer1_w[0][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[0][2] of the 1st layer w13 is :")
        print(self.sess.run(layer1_w[0][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[0][3] of the 1st layer w14 is :")
        print(self.sess.run(layer1_w[0][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[1][0] of the 1st layer w21 is :")
        print(self.sess.run(layer1_w[1][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[1][1] of the 1st layer w22 is :")
        print(self.sess.run(layer1_w[1][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))



        print ("the layer1_w[1][2] of the 1st layer w13 is :")
        print(self.sess.run(layer1_w[1][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[1][3] of the 1st layer w14 is :")
        print(self.sess.run(layer1_w[1][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[2][2] of the 1st layer w22 is :")
        print(self.sess.run(layer1_w[2][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[2][3] of the 1st layer w23 is :")
        print(self.sess.run(layer1_w[2][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][0] of the 1st layer w41 is :")
        print(self.sess.run(layer1_w[3][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][1] of the 1st layer w42 is :")
        print(self.sess.run(layer1_w[3][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][2] of the 2nd layer w43 is :")
        print(self.sess.run(layer1_w[3][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][3] of the 2nd layer w44 is :")
        print(self.sess.run(layer1_w[3][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        


        delta_w11 = -2*(1-Y1)*self.w2[0][0]*x1 + 2*Y2*self.w2[0][1]*x1

        # pratial L / partial w11
             
        


        delta_w11 = -2*(1-Y1)*self.w2[0][0]*(1-self.layer1_output[0][0]*self.layer1_output[0][0])*x1 + 2*Y2*self.w2[0][1]*(1-self.layer1_output[0][0]*self.layer1_output[0][0])*x1
        print ("1-Y1", 1-Y1)
        print ("the delta of the w11 is :")
        print(self.sess.run(delta_w11, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        delta_b1 = -2*(1-Y1)*self.w2[0][0]*(1-self.layer1_output[0][0]*self.layer1_output[0][0]) + 2*Y2*self.w2[0][1]*(1-self.layer1_output[0][0]*self.layer1_output[0][0])
        print ("the delta of the b1 is :")
        print(self.sess.run(delta_b1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
























        ###################### 22222222222222222222222222222222 ############################
        ############ ******************* the second action and step training happens here ***********  ############
        print ("############### now go into the 2nd step cacualtion ###############")
        print ("")
        print ("###############  2nd step cacualtion ###############")

        
        current_step = 1

        # note here we can assume that we already get the array_input, in the c code should store all the input data information
        # take the second input data set from the array_input

        x1 = array_input[current_step][0]
        x2 = array_input[current_step][1]
        x3 = array_input[current_step][2]
        x4 = array_input[current_step][3]
        print ("the current_step input 1 is :", current_step)
        print ("the current_step input 1 is x1:", x1)
        print(self.sess.run(x1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print ("the second input 2 is x2:", x2)
        print(self.sess.run(x2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print ("the second input 3 is x3:", x3)
        print(self.sess.run(x3, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print ("the second input 4 is x4:", x4)
        print(self.sess.run(x4, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        # verify here if the input data is correct


  



        # action_probability = self.all_act_prob
        # take the second action probability here
        Y1 = action_probability[current_step][0]  # the first unit output 
        Y2 = action_probability[current_step][1]  # the second unit output

        print ("the second action prob is Y1:", action_probability[current_step][0])
        print(self.sess.run(Y1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the second action prob is Y2:", action_probability[current_step][1])
        print(self.sess.run(Y2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))



        # note here the weights are not changed 
        print(" the w2 is the first layer weight: ", self.w2)
        print(self.sess.run(self.w2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        weight2 = self.w2
        print ("the weight2[0][0] is:", weight2[0][0])
        print(self.sess.run(weight2[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        weight1 = self.w1
        print ("the weight1[0][0] is:", weight1[0][0])
        print(self.sess.run(weight1[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))



        tanh_delta0 = (1-self.layer1_output[current_step][0]*self.layer1_output[current_step][0])
        tanh_delta1 = (1-self.layer1_output[current_step][1]*self.layer1_output[current_step][1])
        tanh_delta2 = (1-self.layer1_output[current_step][2]*self.layer1_output[current_step][2])
        tanh_delta3 = (1-self.layer1_output[current_step][3]*self.layer1_output[current_step][3])

        print ("the layer1_output (whole) is :", self.layer1_output)
        print(self.sess.run(self.layer1_output, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_output[1][0] is :", self.layer1_output[current_step][0])
        print(self.sess.run(self.layer1_output[current_step][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_output[1][1] is :", self.layer1_output[current_step][1])
        print(self.sess.run(self.layer1_output[current_step][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_output[1][2] is :", self.layer1_output[current_step][2])
        print(self.sess.run(self.layer1_output[current_step][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_output[1][3] is :", self.layer1_output[current_step][3])
        print(self.sess.run(self.layer1_output[current_step][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))




        print ("the tanh_delta is :", tanh_delta0)
        print(self.sess.run(tanh_delta0, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        ############# here start to calculate the diff layer by layer #######################
        diff_start1 = -every_step_reward[current_step]*one_hot_action[current_step][0]   # 5.5203
        diff_start2 = -every_step_reward[current_step]*one_hot_action[current_step][1]   # 0
        

        
        # one_hot_action[0][0]
        # one_hot_action[0][0] one_hot_action[0][1] one_hot_action[1][0] one_hot_action[1][1]   is 1  0  0  1


        print ( " diff-start is :")
        print(self.sess.run(diff_start1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print(self.sess.run(diff_start2, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        """
        print(self.sess.run(one_hot_action[1][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        print(self.sess.run(one_hot_action[1][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        """



        #tf_const_diff_soft = tf.constant([[0.0,0.0,0.0],[0.0,0.0,0.0,],[0.0,0.0,0.0]])

        #diff_soft = tf.get_variable("diff_soft", dtype=tf.float32,initializer=tf_const_diff_soft)
        # here we only define the diff_soft that is the output of the softmax (backpropagation)
        diff_soft = [[0.0, 0.0], [0.0, 0.0]]
        # 13 is the number of steps in one episode


        diff_soft11 = (1-Y1)/13
        diff_soft12 = -Y2/13
        diff_soft21 = -Y1/13
        diff_soft22 = (1-Y2)/13
        # here is the softmax layer inner information to help calculate the backpropogation

        # the diff_start is the input of the backpropagation
        diff_soft[0][0] = diff_soft11 * diff_start1
        diff_soft[0][1] = diff_soft12 * diff_start1
        diff_soft[1][0] = diff_soft21 * diff_start2
        diff_soft[1][1] = diff_soft22 * diff_start2





        
        print (" ")
        print ("*******************")
        print ("the diff_soft is:")
        print(self.sess.run(diff_soft, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        


        
        delta_layer2_w11 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][0]
        delta_layer2_w12 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][0]


        delta_layer2_w21 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][1]
        delta_layer2_w22 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][1]

        delta_layer2_w31 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][2]
        delta_layer2_w32 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][2]


        delta_layer2_w41 = (diff_soft[0][0] + diff_soft[1][0]) * self.layer1_output[0][3]
        delta_layer2_w42 = (diff_soft[0][1] + diff_soft[1][1]) * self.layer1_output[0][3]


        layer2_w = [[0.0, 0.0], [0.0, 0.0],[0.0, 0.0],[0.0, 0.0]]
        layer1_w = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0,0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]]


        # layer2_w[0][1]
        print ("the weight2[0][0] #### before was :")
        
        print(self.sess.run(weight2[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))
        


        
        layer2_w[0][0] = weight2[0][0] - 0.1 * delta_layer2_w11
        layer2_w[0][1] = weight2[0][1] - 0.1 * delta_layer2_w12
        layer2_w[1][0] = weight2[1][0] - 0.1 * delta_layer2_w21
        layer2_w[1][1] = weight2[1][1] - 0.1 * delta_layer2_w22
        layer2_w[2][0] = weight2[2][0] - 0.1 * delta_layer2_w31
        layer2_w[2][1] = weight2[2][1] - 0.1 * delta_layer2_w32
        layer2_w[3][0] = weight2[3][0] - 0.1 * delta_layer2_w41
        layer2_w[3][1] = weight2[3][1] - 0.1 * delta_layer2_w42
        

        
        # please see the note book, the first example for w11 in the first layer
        delta_layer1_w11[1] =  ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x1
        delta_layer1_w12[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x1
        delta_layer1_w13[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x1
        delta_layer1_w14[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x1



        delta_layer1_w21[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x2
        #  actually the w21 is more close to the w11, we should keep them together     

        delta_layer1_w22[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x2
        delta_layer1_w23[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x2
        delta_layer1_w24[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x2



        delta_layer1_w31[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x3
        delta_layer1_w32[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x3
        delta_layer1_w33[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x3
        delta_layer1_w34[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x3


        delta_layer1_w41[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[0][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[0][1])*tanh_delta0*x4
        delta_layer1_w42[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[1][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[1][1])*tanh_delta1*x4
        delta_layer1_w43[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[2][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[2][1])*tanh_delta2*x4
        delta_layer1_w44[1]  = ((diff_soft[0][0] + diff_soft[1][0]) * weight2[3][0] +  (diff_soft[0][1] + diff_soft[1][1]) * weight2[3][1])*tanh_delta3*x4


        
        ####################### here we are inside the second step  ########################
        # change the step num here to 13 in near future
        step_num = 2

        print ("***********************  in the second step updating weights  ******************")


        # clear the total delta, recaculate in each step
        delta_layer1_total_w11 = 0.0
        delta_layer1_total_w12 = 0.0
        delta_layer1_total_w13 = 0.0
        delta_layer1_total_w14 = 0.0

        delta_layer1_total_w21 = 0.0
        delta_layer1_total_w22 = 0.0
        delta_layer1_total_w23 = 0.0
        delta_layer1_total_w24 = 0.0

        delta_layer1_total_w31 = 0.0
        delta_layer1_total_w32 = 0.0
        delta_layer1_total_w33 = 0.0
        delta_layer1_total_w34 = 0.0

        delta_layer1_total_w41 = 0.0
        delta_layer1_total_w42 = 0.0
        delta_layer1_total_w43 = 0.0
        delta_layer1_total_w44 = 0.0
        
        # here set to one only for the for loop test

        print ("here we are in the second step")
        # the range for loop from the 0 to step_num-1 
        for num in range(0,step_num):
            print ("num is", num)
        # the for loop will operate from 0 to 1    

        for num in range(0,step_num):
            delta_layer1_total_w11 = delta_layer1_total_w11 + delta_layer1_w11[num]
            print ("the delta_layer1_w11[num] of the 1st layer w11 is :", num)
            print(self.sess.run(delta_layer1_w11[num] , feed_dict={    # train_op is the Adam Optimizer
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,
            }))

        for num in range(0,step_num):
            delta_layer1_total_w12 = delta_layer1_total_w12 + delta_layer1_w12[num]
            print ("the delta_layer1_w12[num] of the 1st layer w12 is :", num)
            print(self.sess.run(delta_layer1_w12[num] , feed_dict={    # train_op is the Adam Optimizer
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,
            }))
    
        for num in range(0,step_num):
            delta_layer1_total_w13 = delta_layer1_total_w13 + delta_layer1_w13[num]
            print ("the delta_layer1_w13[num] of the 1st layer w13 is :", num)
            print(self.sess.run(delta_layer1_w13[num] , feed_dict={    # train_op is the Adam Optimizer
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,
            }))

        for num in range(0,step_num):
            delta_layer1_total_w14 = delta_layer1_total_w14 + delta_layer1_w14[num]
            print ("the delta_layer1_w14[num] of the 1st layer w14 is :", num)
            print(self.sess.run(delta_layer1_w14[num] , feed_dict={    # train_op is the Adam Optimizer
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,
            }))


        print ("the delta_layer1_total_w11 of the 1st layer w11 is :")
        print(self.sess.run(delta_layer1_total_w11 , feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the delta_layer1_total_w12 of the 1st layer w12 is :")
        print(self.sess.run(delta_layer1_total_w12 , feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the delta_layer1_total_w13 of the 1st layer w13 is :")
        print(self.sess.run(delta_layer1_total_w13 , feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the delta_layer1_total_w14 of the 1st layer w14 is :")
        print(self.sess.run(delta_layer1_total_w14 , feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))






        for num in range(0,step_num):
            delta_layer1_total_w21 = delta_layer1_total_w21 + delta_layer1_w21[num] 

        for num in range(0,step_num):
            delta_layer1_total_w22 = delta_layer1_total_w22 + delta_layer1_w22[num]
    
        for num in range(0,step_num):
            delta_layer1_total_w23 = delta_layer1_total_w23 + delta_layer1_w23[num]

        for num in range(0,step_num):
            delta_layer1_total_w24 = delta_layer1_total_w24 + delta_layer1_w24[num]




        for num in range(0,step_num):
            delta_layer1_total_w31 = delta_layer1_total_w31 + delta_layer1_w31[num] 

        for num in range(0,step_num):
            delta_layer1_total_w32 = delta_layer1_total_w32 + delta_layer1_w32[num]
    
        for num in range(0,step_num):
            delta_layer1_total_w33 = delta_layer1_total_w33 + delta_layer1_w33[num]

        for num in range(0,step_num):
            delta_layer1_total_w34 = delta_layer1_total_w34 + delta_layer1_w34[num]


        for num in range(0,step_num):
            delta_layer1_total_w41 = delta_layer1_total_w41 + delta_layer1_w41[num] 

        for num in range(0,step_num):
            delta_layer1_total_w42 = delta_layer1_total_w42 + delta_layer1_w42[num]
    
        for num in range(0,step_num):
            delta_layer1_total_w43 = delta_layer1_total_w43 + delta_layer1_w43[num]

        for num in range(0,step_num):
            delta_layer1_total_w44 = delta_layer1_total_w44 + delta_layer1_w44[num]




        layer1_w[0][0] = weight1[0][0] - 0.1 * delta_layer1_total_w11
        layer1_w[0][1] = weight1[0][1] - 0.1 * delta_layer1_total_w12
        layer1_w[0][2] = weight1[0][2] - 0.1 * delta_layer1_total_w13
        layer1_w[0][3] = weight1[0][3] - 0.1 * delta_layer1_total_w14

        layer1_w[1][0] = weight1[1][0] - 0.1 * delta_layer1_total_w21
        layer1_w[1][1] = weight1[1][1] - 0.1 * delta_layer1_total_w22
        layer1_w[1][2] = weight1[1][2] - 0.1 * delta_layer1_total_w23
        layer1_w[1][3] = weight1[1][3] - 0.1 * delta_layer1_total_w24

        layer1_w[2][0] = weight1[2][0] - 0.1 * delta_layer1_total_w31
        layer1_w[2][1] = weight1[2][1] - 0.1 * delta_layer1_total_w32
        layer1_w[2][2] = weight1[2][2] - 0.1 * delta_layer1_total_w33
        layer1_w[2][3] = weight1[2][3] - 0.1 * delta_layer1_total_w34


        layer1_w[3][0] = weight1[3][0] - 0.1 * delta_layer1_total_w41
        layer1_w[3][1] = weight1[3][1] - 0.1 * delta_layer1_total_w42
        layer1_w[3][2] = weight1[3][2] - 0.1 * delta_layer1_total_w43
        layer1_w[3][3] = weight1[3][3] - 0.1 * delta_layer1_total_w44




        print ("here we are in the second step")
        print ("the layer1_w[0][0] of the 1st layer w11 is :")
        print(self.sess.run(layer1_w[0][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[0][1] of the 1st layer w12 is :")
        print(self.sess.run(layer1_w[0][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[0][2] of the 1st layer w13 is :")
        print(self.sess.run(layer1_w[0][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[0][3] of the 1st layer w14 is :")
        print(self.sess.run(layer1_w[0][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[1][0] of the 1st layer w21 is :")
        print(self.sess.run(layer1_w[1][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[1][1] of the 1st layer w22 is :")
        print(self.sess.run(layer1_w[1][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))



        print ("the layer1_w[1][2] of the 1st layer w13 is :")
        print(self.sess.run(layer1_w[1][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[1][3] of the 1st layer w14 is :")
        print(self.sess.run(layer1_w[1][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[2][2] of the 1st layer w22 is :")
        print(self.sess.run(layer1_w[2][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))

        print ("the layer1_w[2][3] of the 1st layer w23 is :")
        print(self.sess.run(layer1_w[2][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][0] of the 1st layer w41 is :")
        print(self.sess.run(layer1_w[3][0], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][1] of the 1st layer w42 is :")
        print(self.sess.run(layer1_w[3][1], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][2] of the 2nd layer w43 is :")
        print(self.sess.run(layer1_w[3][2], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("the layer1_w[3][3] of the 2nd layer w44 is :")
        print(self.sess.run(layer1_w[3][3], feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        delta_w11 = -2*(1-Y1)*self.w2[0][0]*x1 + 2*Y2*self.w2[0][1]*x1

        # pratial L / partial w11
             
        


        delta_w11 = -2*(1-Y1)*self.w2[0][0]*(1-self.layer1_output[0][0]*self.layer1_output[0][0])*x1 + 2*Y2*self.w2[0][1]*(1-self.layer1_output[0][0]*self.layer1_output[0][0])*x1
        print ("1-Y1", 1-Y1)
        print ("the delta of the w11 is :")
        print(self.sess.run(delta_w11, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        delta_b1 = -2*(1-Y1)*self.w2[0][0]*(1-self.layer1_output[0][0]*self.layer1_output[0][0]) + 2*Y2*self.w2[0][1]*(1-self.layer1_output[0][0]*self.layer1_output[0][0])
        print ("the delta of the b1 is :")
        print(self.sess.run(delta_b1, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))











































        ###################### before was our self update test ###################################


        ########################### the real training happens here ###################################

        self.sess.run(self.train_op, feed_dict={    # train_op is the ASdam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })







        # HERE try to only take the first action and see.   
        
        print("")
        print("all the parameters after we run sess run train.op")
        #################  the debugging signals ################################
        print ("np.shape(self.ep_as)", np.shape(self.ep_as))
        print ("np.shape(self.tf_acts)", np.shape(self.tf_acts))
        print ("np.shape(self.tf_vt)", np.shape(self.tf_vt))
        # print ("np.array(self.ep_as)", np.array(self.ep_as))
        # [0 1 1 0 1 0 1] , shows the actions that have taken

        ######### uncommit below,  here below for debug
        
        ###################  the six inportant signals ##########################
        tensor_tf_acts = self.tf_acts
        one_hot_action = tf.one_hot(self.tf_acts, self.n_actions)
        action_probability = self.all_act_prob
        every_step_reward = self.tf_vt

        print ("self.tf_obs = np.vstack(self.ep_obs, the input is :")
        print(self.sess.run(self.tf_obs, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        print ("self.tf_acts = the action takes :")
        print(self.sess.run(self.tf_acts, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        print ("self.tf_vt = the reword gets :")
        print(self.sess.run(self.tf_vt, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))
        

        print ("self.tf_acts = np.array(self.ep_as) is :")
        print(self.sess.run(tensor_tf_acts, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))
        # [0 1 1 1 0 0 1 1 0 0 1 0 1 0 0 0 0 0 1]
        # self.tf_acts stores all the actions that have taken



        print ("one_hot_action = tf.one_hot(self.tf_acts, self.n_actions):")
        print(self.sess.run(one_hot_action, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        print ("action_probability = tf.all_act_prob: ")
        print(self.sess.run(action_probability, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        print ("every_step_reward = tf.vt: ")
        print(self.sess.run(every_step_reward, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        log_all_act_prob = tf.log(self.all_act_prob)
        print ("log_all_act_prob = log (tf.all_act_prob) : ")
        print(self.sess.run(log_all_act_prob, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))


        prob_mult_one_hot = tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions)
        print ("prob_mult_one_hot = tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions) : ")
        print(self.sess.run(prob_mult_one_hot, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))




        reduce_sum_debug = tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        print ("reduce_sum_debug = neg_log_prob = tf.reduce_sum : ")
        print(self.sess.run(reduce_sum_debug, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        neg_log_prob = tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions), axis=1)

        print ("the neg_log_prob is tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions )")
        print(self.sess.run(neg_log_prob, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        multiply_ = neg_log_prob * self.tf_vt
        print ("the multiply_ multiply_ = neg_log_prob * self.tf_vt ")
        print(self.sess.run(multiply_, feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))




        print ("the total loss is tf.reduce_mean(neg_log_prob * self.tf_vt) ")
        print(self.sess.run(tf.reduce_mean(neg_log_prob * self.tf_vt), feed_dict={    # train_op is the Adam Optimizer
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,
        }))


        print ("############### end of the debugging ##############################")
        ############### end of the debugging ##############################
        



        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        print ("###########################")
        print ("now in the _discount_and_norm_rewards function")
        # first we needs to know the size of the ep_rs
        print ("")
        # print ("the self.ep_rs is", self.ep_rs)
        print ("")
        print ("the shape of the nparray.ep_rs is", np.shape(self.ep_rs))

        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # print ("reversed(range(0, len(self.ep_rs) is the for loop sequence :", list(reversed(range(0, len(self.ep_rs)))))

        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
            print ("the current discounted_ep_rs[",t,"] is", running_add)

        # here this for loop first caculate the  discounted_ep_rs[36] = 1
        # then it caculate the  discounted_ep_rs[35] = 1 * 0.95 + 1 = 1.99
        # then is caculate the  discounted_ep_rs[34] = 1.97 * 0.95 + 1 = 2.97




        # normalize episode rewards
        # comment here for a hard train
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        #print ("np.std(discounted_ep_rs) is ", np.std(discounted_ep_rs))
        # if only have one step, here the std of the array is 0, because there is no diviation
        #discounted_ep_rs /= np.std(discounted_ep_rs)
        
        ################ debug signals for discounted reward ##########################
        print ("")
        print ("the _discount_and_norm_rewards will return the discount reward")
 
        for t in reversed(range(0, len(self.ep_rs))):
            print ("the current discounted_ep_rs[",t,"] is", discounted_ep_rs[t])


        # discounted_ep_rs[0] = 2
        # now the 1 is returned for a hard train
        discounted_ep_rs = [5.5203495,  4.6339645, 0,0,0,  0,0,0,0,0,   0,0,0]
        # discounted_ep_rs = [0,  4.6339645, 0,0,0,  0,0,0,0,0,   0,0,0]
        # discounted_ep_rs = [5.5203495, 0 ,0, 0, 0,  0,0,0,0,0,   0,0,0]
        # [ 5.5203495   4.6339645   3.7386262   2.8342443   1.920727    0.99798226   0.06591693 -0.8755632  -1.8265532  -2.7871492  -3.7574482  -4.7375484 -5.727548  ]


        return discounted_ep_rs

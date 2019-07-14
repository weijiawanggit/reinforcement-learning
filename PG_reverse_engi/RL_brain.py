"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.
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
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=4,
            # the structure changed here
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   
            # this is negative log of chosen action
            
            # or in this way:
            # this way is the recommended way and more clear
            neg_log_prob = tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            
            
            # added here for debug
            # with tf.Session() as sess:
            #    print sess.run(x)

            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

            ################ data debug  #####################
            # multiply test for debug, not influence the main process
            multiply = neg_log_prob * self.tf_vt
            print ("the shape of the multiply is", np.shape(multiply))




        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)



# the way to print value in tensorflow
#    with tf.Session():
  #    print(c.eval())

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # print "now in the learn function, the self.ep_obs is", self.ep_obs
        # print ""
        print ("the shape of the nparray.ep_obs is", np.shape(self.ep_obs))
        # train on one complete episode
        self.sess.run(self.train_op, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        print ("in the learning process")
        

        print ("np.shape(self.ep_as)", np.shape(self.ep_as))
        print ("np.array(self.ep_as)", np.array(self.ep_as))

        # here below for debug
        
        tensor_tf_acts = self.tf_acts
        one_hot_action = tf.one_hot(self.tf_acts, self.n_actions)
        action_probability = self.all_act_prob
        every_step_reward = self.tf_vt

        log_all_act_prob = tf.log(self.all_act_prob)
        reduce_sum_debug = tf.reduce_sum(  -tf.log(self.all_act_prob)  *   tf.one_hot(self.tf_acts, self.n_actions), axis=1)


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



        print ("log_all_act_prob = log (tf.all_act_prob) : ")
        print(self.sess.run(log_all_act_prob, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        print ("reduce_sum_debug = neg_log_prob = tf.reduce_sum : ")
        print(self.sess.run(reduce_sum_debug, feed_dict={    # train_op is the Adam Optimizer
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))







        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

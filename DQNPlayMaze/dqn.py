import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter#每隔多少步更新targetnet的参数
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory_counter=0#初始化当前的记忆长度

        # total learning step
        self.learn_step_counter = 0

        self.memory=np.zeros((self.memory_size,n_features*2+2))
        self.build_net()

        t_param=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target_net') #暂时冻结，用于评判r_nextstate
        e_param=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='eval_net')  #实时更新,用于评判r_thisstate

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op=[tf.assign(t,e) for t,e in zip(t_param,e_param)] #将eval_net中的参数更新到targetnet中

        self.sess=tf.Session()

        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
        self.cost_his=[]


    """
    创建神经网络
    """
    def build_net(self):
        self.s=tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.s_=tf.placeholder(tf.float32,[None,self.n_features],name='s_')
        self.r=tf.placeholder(tf.float32,[None,],name='r')
        self.a=tf.placeholder(tf.int32,[None],name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            e1=tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            self.q_eval=tf.layers.dense(e1,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='q')

        with tf.variable_scope('target_net'):
            t1=tf.layers.dense(self.s_,20,tf.nn.relu)
            self.q_next=tf.layers.dense(t1,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices=tf.stack([tf.range(tf.shape(self.a)[0],dtype=tf.int32),self.a],axis=1)#合并序列和a的index，如[[1,3],[2,1]....]
            self.q_eval_wrt_a=tf.gather_nd(params=self.q_eval,indices=a_indices)#根据a_indices获取q_eval中对应的神经网路预测

        with tf.variable_scope('loss'):
            self.loss=tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_wrt_a,name='TD_error'))
        with tf.variable_scope('train'):
            self.train_op=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    """
    存储记忆，最大长度限制self.memory_size
    """
    def store_transition(self,s,a,r,s_):
        transition=np.hstack((s,[a,r],s_))#存储记忆s,a,r,s_当前状态，动作，下一状态
        index=self.memory_counter%self.memory_size
        self.memory[index,:]=transition
        self.memory_counter+=1

    """
    选择动作，给与一定的概率随机选择一个动作，以达到探索的目的
    """
    def choose_action(self,observation):
        #因为神经网络的输入是[[None,.state]],故需要增加一个维度
        observation=observation[np.newaxis,:]
        #得到预测的四个动作的奖励/概率,self.epsilon=探索几率
        if np.random.uniform()<self.epsilon:
            actions_value=self.sess.run(self.q_eval,{self.s:observation})
            action=np.argmax(actions_value)#从预测中选择奖励最大的动作
            # print('observation',observation,'\taction_value:', actions_value, '\t action:', action)
        else:
            action =np.random.choice(np.arange(self.n_actions))
        return action

    """
    从记忆库中随机选择一批动作来进行更新q_eval的更新
    """
    def learn(self):
        #进行q_eval与q_targetnet中的参数替换，更新q_targetnet中的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
        sample_index=np.random.choice(np.arange(self.memory.shape[0]),size=self.batch_size)
        batch_memory=self.memory[sample_index]

        _,loss_=self.sess.run([self.train_op,self.loss],{self.s:batch_memory[:,:self.n_features],
                                                         self.a:batch_memory[:,self.n_features],
                                                         self.r:batch_memory[:,self.n_features+1],
                                                         self.s_:batch_memory[:,-self.n_features:]})

        self.cost_his.append(loss_)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter+=1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph=True)
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

np.random.seed(1)
tf.set_random_seed(1)

"""
使用DQN来玩飞翔的小鸟，
state=当前的四帧图像=[image1,image2,image3,image4]，原始228*512*3->转化为80*80*1
a=[up,down]
r=0.1 正常，1通过柱子 ，-1装上柱子
terminal：false，true
"""

class DQNNetwork:

    def __init__(self,lr=0.01,e_greedy=0.9,gamma=0.99,memory_size=50000,replace_iters=100):
        self.lr = lr # 学习率
        self.e_greedy=e_greedy#选择最大奖励动作的概率
        self.gamma=gamma#奖励的衰减率
        self.memory_size=memory_size#记忆库大小
        self.replace_iters=replace_iters#每训练replace_iters次之后将evalnet的参数复制到tergetnet中
        self.memory=pd.DataFrame(columns=['state','a','r','state_next'])
        self.memory_counter=0#store记忆的次数
        self.learn_step_counter = 0#已更新eval_net次数
        self.batch_size=128

        self.build_net()
        t_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')  # 暂时冻结，用于评判r_nextstate
        e_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')  # 实时更新,用于评判r_thisstate

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_param, e_param)]  # 将eval_net中的参数更新到targetnet中

        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    #创建eval_net & target_net
    def build_net(self):
        self.state=tf.placeholder(tf.float32,[None,80,80,4],name='state')#环境传来的图像，4帧合并为1个state=[前三帧，当前action执行后返回的帧]
        self.state_next=tf.placeholder(tf.float32,[None,80,80,4],name='state_next')
        self.a=tf.placeholder(tf.int32,[None,],name='a')# 传入动作的index
        self.r=tf.placeholder(tf.float32,[None,],name='r')#环境的奖励

        #evalnet,
        with tf.variable_scope('eval_net'):
            eval_conv1=tf.layers.conv2d(self.state,32,7,strides=2,padding='same',activation=tf.nn.relu,name='eval_conv1')#40*40*32
            eval_pool1=tf.layers.max_pooling2d(eval_conv1,2,2,name='eval_pool1')#20*20*32

            eval_conv2 = tf.layers.conv2d(eval_pool1, 64, 5, strides=2, padding='same', activation=tf.nn.relu,name='eval_conv2')#10*10*64
            eval_pool2=tf.layers.max_pooling2d(eval_conv2,2,2,padding='same',name='eval_pool2')#5*5*64

            eval_conv3 = tf.layers.conv2d(eval_pool2, 64, 3, strides=1, padding='same', activation=tf.nn.relu,name='eval_conv3')  # 5*5*64
            eval_pool3 = tf.layers.max_pooling2d(eval_conv3, 2, 2, padding='same',name='eval_pool3')  # 3*3*64

            eval_flat=tf.reshape(eval_pool3,[-1,3*3*64],name='eval_flat')
            eval_l1=tf.layers.dense(eval_flat,32,activation=tf.nn.relu,name='eval_l1')
            self.eval_output=tf.layers.dense(eval_l1,2, activation=tf.nn.softmax,name='eval_output')#输出动作概率[0.9,0.1]


        #targetnet
        with tf.variable_scope("target_net"):
            target_conv1 = tf.layers.conv2d(self.state_next, 32, 7, strides=2, padding='same', activation=tf.nn.relu,name='target_conv1')  # 20*20*32
            target_pool1 = tf.layers.max_pooling2d(target_conv1, 2, 2, name='target_pool1')  # 10*10*32

            target_conv2 = tf.layers.conv2d(target_pool1, 64, 5, strides=2, padding='same', activation=tf.nn.relu,name='target_conv2')  # 5*5*64
            target_pool2 = tf.layers.max_pooling2d(target_conv2, 2, 2, padding='same', name='target_pool2')  # 3*3*64

            target_conv3 = tf.layers.conv2d(target_pool2, 64, 3, strides=1, padding='same', activation=tf.nn.relu,name='target_conv3')  # 3*3*64
            target_pool3 = tf.layers.max_pooling2d(target_conv3, 2, 2, padding='same', name='target_pool3')  # 2*2*64

            target_flat = tf.reshape(target_pool3, [-1, 3 * 3 * 64], name='target_flat')
            target_l1 = tf.layers.dense(target_flat, 32, activation=tf.nn.relu, name='target_l1')
            self.target_output = tf.layers.dense(target_l1, 2, activation=tf.nn.softmax,name='target_output')  # 输出动作概率[0.9,0.1]

        #根据DQN计算state_next的奖励,并冻结targetnet的参数
        with tf.variable_scope('q_target_reward'):
            q_target_reward = self.r + self.gamma * tf.reduce_max(self.target_output, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target_reward = tf.stop_gradient(q_target_reward)

        #根据输入的a，找到eval_net的输出，然后计算loss
        with tf.variable_scope('q_eval_reward'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],axis=1)  # 合并序列和a的index，如[[1,3],[2,1]....]
            self.q_eval_reward_a = tf.gather_nd(params=self.eval_output, indices=a_indices)  # 根据a_indices获取q_eval中对应的神经网路预测

        #损失函数
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval_reward_a, self.q_target_reward, name='TD_error'))
        with tf.variable_scope('train'):
            self.train_op=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    """
    存储训练记忆
    """
    def store_memory(self,state,a,r,state_next):
        index = self.memory_counter % self.memory_size
        self.memory.loc[index]={
            'state':state,
            'a':a,
            'r':r,
            'state_next':state_next
        }
        self.memory_counter+=1


    """
    选择一个动作，根据state获取网络的输出
    """
    def choose_action(self,state):
        state=state[np.newaxis,:]#增加一个维度
        action = [0, 0]
        action_reward=np.array([0.5,0.5])
        if np.random.uniform()<self.e_greedy:
            action_reward=self.sess.run(self.eval_output,{self.state:state})
            action_arg=np.argmax(action_reward)
            action[action_arg]=1
        else:
            action_arg=np.random.randint(0,2)
            action[action_arg]=1

        return action,action_reward.round(2)

    """
    从记忆库中选择一批状态进行更新eval
    """
    def learn(self):

        #一定间隔更新target_net
        if self.learn_step_counter%self.replace_iters==0:
            self.sess.run(self.target_replace_op)
            print("replace params,learn_step_counter:",self.learn_step_counter)

        #每一万步保存一次模型
        if self.learn_step_counter>100 and self.learn_step_counter%10000==0:
            saver=tf.train.Saver()
            print("保存模型：step-",self.learn_step_counter)
            saver.save(self.sess,'./birdModel.ckpt')

        sample_index=np.random.choice(np.arange(self.memory.shape[0]),size=self.batch_size)
        # batch_memory = self.memory.loc[sample_index]

        state=np.array([ s for s in self.memory.loc[sample_index,'state'].values])
        a = np.array([np.array(s).argmax() for s in self.memory.loc[sample_index, 'a'].values])
        r = np.array([s for s in self.memory.loc[sample_index,'r'].values])
        state_next = np.array([s for s in self.memory.loc[sample_index,'state_next'].values])

        _,loss_=self.sess.run([self.train_op,self.loss],{self.state:state,
                                                         self.a:a,
                                                         self.r:r,
                                                         self.state_next:state_next})
        self.learn_step_counter += 1


if __name__ == '__main__':
    DQN = DQNNetwork()
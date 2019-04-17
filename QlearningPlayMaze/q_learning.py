import numpy as np
import pandas as pd

"""
1.choose_action, 选择动作
2.learn, 更新Q_table （记忆）
"""
class Q_Learning:
    def __init__(self,actions=['u','d','l','r'],learning_rate=0.01, reward_decay=0.9,epsilon =0.9):
        self.actions=actions#可执行的动作空间
        self.Q=pd.DataFrame(columns=actions,dtype=np.float32)#Q表
        self.lr = learning_rate#学习率
        self.gamma = reward_decay#衰减
        self.epsilon = epsilon#随机概率

    """
    根据Q表最大奖励的动作s和随机选择动作->确定下一步执行的动作
    """
    def choose_action(self,state):
        self.check_state_exist(state)#当前state是否存在，一个新的state需要添加到Q中

        #90%的情况按Qtable进行选择
        if np.random.uniform(0,1)<self.epsilon:
            actions_reward=self.Q.loc[state]
            actions=actions_reward[actions_reward==actions_reward.max()].index
            action=np.random.choice(actions)
        else:
            action=np.random.choice(self.actions)

        return action

    """
    检查state在Qtable中是否存在，不存在则创建全零
    """
    def check_state_exist(self,state):
        #如果不存在则初始化为全零
        if state not in self.Q.index:
            self.Q.loc[state]=np.zeros(shape=len(self.actions),dtype=np.float32)


    #更新Qtable
    def learn(self,state,action,reward,state_next):
        self.check_state_exist(state_next)

        #如果为结束标志，更新[state,action]=reward
        if state_next=='terminal':
            self.Q.loc[state,action]=reward

        #根据公式进行更新
        else:
            self.Q.loc[state,action]+=self.lr*(reward+ self.gamma*(self.Q.loc[state_next,:].max()-self.Q.loc[state,action]))

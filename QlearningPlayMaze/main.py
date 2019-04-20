from maza.maze import Maze
from QlearningPlayMaze.q_learning import Q_Learning

Env=Maze()
Agent=Q_Learning(actions=Env.action_space)

def train(iter=100):
    for i in range(iter):
        print("iter:",i)
        #每次玩之前先reset游戏
        state=Env.reset()
        while True:
            Env.render()
            action=Agent.choose_action(str(state))#选择动作
            state_next,reward,done=Env.step(action)#步进环境获取反馈
            Agent.learn(str(state),action,reward,str(state_next))#更新Agent
            state=state_next

            #如果游戏终止则跳出
            if done:
                break;

if __name__=="__main__":
    Env.after(1,train)
    Env.mainloop()
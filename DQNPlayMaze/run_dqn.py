from maza.maze_dqn import Maze
from DQNPlayMaze.dqn import DeepQNetwork

Env=Maze()
Agent= DeepQNetwork(Env.n_actions, Env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      )

def train():
    step = 0  # 记录走行总步数
    for i in range(300):
        observation=Env.reset()
        while True:
            Env.render()
            action=Agent.choose_action(observation)
            observation_,reward,done=Env.step(action)
            Agent.store_transition(observation,action,reward,observation_)

            #每5步更新q_eval的误差
            if step>200 and step%5==0:
                Agent.learn()

            if done:
                break
            step+=1
            observation=observation_
        print('iter:', i,'\tstep:',step)

    print("game over")


if __name__=='__main__':
    Env.after(100,train)
    Env.mainloop()
    Agent.plot_cost()

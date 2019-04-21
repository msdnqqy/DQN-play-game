import  game.wrapped_flappy_bird  as game
import cv2
from DQNPlayBird.dqn import DQNNetwork
import numpy as np

ENV = game.GameState()
Agent=DQNNetwork()

def train(iter=1000):
    # x_t, r_0, terminal = ENV.frame_step(do_nothing)
    #初始化，当不存在记忆时候，复制4帧
    step=0
    for i in range(iter):
        while True:
            if Agent.memory_counter==0:
                image_ori, reward, terminal= ENV.frame_step([0,1])
                image=cover_to_gray(image_ori)
                state=np.stack((image,image,image,image),axis=2)

            # print(state.shape)
            action,action_reward=Agent.choose_action(state)
            print('动作：',action,'\taction_p:',action_reward,'\treward:',reward)
            image_ori, reward, terminal = ENV.frame_step(action)
            image = cover_to_gray(image_ori)
            image=np.reshape(image,[80,80,1])

            state_next=np.append(image, state[:, :, :3], axis=2)#[新，旧,旧,旧]
            Agent.store_memory(state,action,reward,state_next)

            # if step>200 and step%5==0:
            Agent.learn()

            state[:,:,:]=state_next[:,:,:]
            step+=1

            #完成一次训练
            if terminal==True:
                break;



#转化为灰度图片
def cover_to_gray(image):
    image_return = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, image_return = cv2.threshold(image_return, 1, 255, cv2.THRESH_BINARY)
    # print(image_return.shape)
    cv2.imshow('inout image',image_return)
    cv2.waitKey(1)
    return image_return


if __name__=='__main__':
    train(100000)


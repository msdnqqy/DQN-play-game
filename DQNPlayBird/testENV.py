import  game.wrapped_flappy_bird  as game
import time
ENV = game.GameState()
do_nothing=[1,0]
x_t, r_0, terminal = ENV.frame_step(do_nothing)
print('donothing:',x_t, r_0, terminal)

for i in range(300):
    up=[0,1]
    x_t, r_0, terminal = ENV.frame_step(up)
    print("up:",x_t.shape, r_0, terminal)
    # time.sleep(0.1)
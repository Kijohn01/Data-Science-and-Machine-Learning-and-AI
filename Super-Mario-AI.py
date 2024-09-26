from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
def trial(instruction_set, render=False, fps=60, random=False):
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.reset()
running_reward = 0
for action in instruction_set:
start_time = time.time()
frame_time = 1.0 / fps # Calculate the time per frame based on FPS
# env.render(); Adding this line would show the attempts
if(render):
env.render()
# of the agent in a pop up window.
# Take random action
if(random):
action = env.action_space.sample()
# Apply the sampled action in our environment
_, reward, _, _ = env.step(action)
running_reward += reward
# Calculate how long to wait to maintain the desired FPS
elapsed_time = time.time() - start_time
time_to_wait = frame_time - elapsed_time
if time_to_wait > 0:
time.sleep(time_to_wait)
return running_reward
if(__name__ == '__main__'):
trial(list(range(1000)), render=True, random=True)

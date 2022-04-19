import copy

from game import BirdEnv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from bird_dqn_agent import DQN
import cv2

SAVE_FREQ = 20
LOAD_MODEL = True
MODEL_PATH = 'model.pth'


def process_image(image_data):
    x_t1 = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    return x_t1


bird = DQN()
env = BirdEnv.BirdEnv()

if LOAD_MODEL:
    bird.load_model(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

for i_episode in range(400):
    s = env.reset()
    img = s[0]
    img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    obs = np.stack((img, img, img, img), axis=2)
    obs = np.transpose(obs, (2, 0, 1))
    ep_r = 0
    while True:
        env.render()
        a = bird.choose_action(obs)
        a_ = [copy.deepcopy(a)]
        a_ = np.array(a_)
        s_, reward, isOver, _ = env.step(a)
        s_ = process_image(s_)
        s_ = np.transpose(s_, (2, 0, 1))
        reward_ = [copy.deepcopy(reward)]
        reward_ = np.array(reward_)
        cur_obs = copy.deepcopy(obs)
        next_obs = copy.deepcopy(s_)
        next_obs = np.append(next_obs, cur_obs[:3, :, :], axis=0)
        r = reward
        ep_r += r
        bird.store_transition(cur_obs, a, r, next_obs)

        if bird.memory_counter > bird.MEMORY_CAPACITY:
            bird.learn()
            if isOver:
                print('Ep:', i_episode, '|', 'Ep_r:', round(ep_r, 2))
        else:
            print("Collecting experience...", bird.memory_counter)
        if isOver:
            break
        obs = next_obs

    # Save model
    if i_episode % SAVE_FREQ == 0:
        bird.save_model(MODEL_PATH)
        print(f"Saving model to {MODEL_PATH}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy
from game import BirdEnv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from bird_dqn_agent import DQN
import cv2


class NetPolicyGradient(nn.Module):
    def __init__(self, N_ACTIONS):
        super(NetPolicyGradient, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(256, N_ACTIONS),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        actions_value = self.out(x)

        return actions_value


class PolicyGradient(object):
    def __init__(self):
        self.LR = 0.01  # learning rate
        self.GAMMA = 0.95  # discount factor
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.N_ACTIONS = 2
        self.net = NetPolicyGradient(2).to(self.DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.step = 0

    def choose_action(self, obs):
        obs = obs[np.newaxis, :, :, :]
        obs = torch.FloatTensor(obs).to(self.DEVICE)
        act_prob = self.net.forward(obs).cpu().detach().view(-1).numpy()
        action = np.random.choice(self.N_ACTIONS, p=act_prob)
        return action

    def store_transition(self, state, action, reward):
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)

    def calc_reward_to_go(self):
        for i in range(len(self.reward_pool) - 2, -1, -1):
            self.reward_pool[i] += self.GAMMA * self.reward_pool[i + 1]
        return np.array(self.reward_pool)

    def learn(self):
        self.step += 1

        discounted_reward = self.calc_reward_to_go()
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)
        discounted_reward = torch.FloatTensor(discounted_reward).to(self.DEVICE)

        state_pool_tensor = torch.FloatTensor(self.state_pool).to(self.DEVICE)
        action_pool_tensor = torch.LongTensor(self.action_pool).to(self.DEVICE)

        act_prob = self.net.forward(state_pool_tensor) + 0.0001 # avoid prob==0
        log_prob = torch.sum(
            -1.0 * torch.log(act_prob) * F.one_hot(action_pool_tensor, act_prob.shape[1]),
            dim=1)
        loss = log_prob * discounted_reward
        loss = torch.mean(loss)
        print('loss: ', loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []


def process_image(image_data):
    x_t1 = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    return x_t1


N_EPISODE = 1000
STEP = 300
TEST = 10


def run():
    bird = PolicyGradient()
    env = BirdEnv.BirdEnv()

    for i_episode in range(1000):
        s = env.reset()
        img = s[0]
        img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        obs = np.stack((img, img, img, img), axis=2)
        obs = np.transpose(obs, (2, 0, 1))
        ep_r = 0
        for step in range(STEP):
            env.render()
            a = bird.choose_action(obs)
            s_, reward, over, _ = env.step(a)
            s_ = process_image(s_)
            s_ = np.transpose(s_, (2, 0, 1))
            cur_obs = copy.deepcopy(obs)
            next_obs = copy.deepcopy(s_)
            next_obs = np.append(next_obs, cur_obs[:3, :, :], axis=0)
            bird.store_transition(cur_obs, a, reward)
            obs = next_obs
            if over:
                bird.learn()
                break
        if i_episode % 5 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                img = state[0]
                img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
                ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
                obs = np.stack((img, img, img, img), axis=2)
                obs = np.transpose(obs, (2, 0, 1))
                for j in range(STEP):
                    env.render()
                    action = bird.choose_action(obs)
                    state, reward, over, _ = env.step(action)
                    s_ = process_image(state)
                    s_ = np.transpose(s_, (2, 0, 1))
                    cur_obs = copy.deepcopy(obs)
                    next_obs = copy.deepcopy(s_)
                    next_obs = np.append(next_obs, cur_obs[:3, :, :], axis=0)
                    total_reward += reward
                    obs = next_obs
                    if over:
                        break
            avg_reward = total_reward / TEST
            print('Ep:', i_episode, '|', 'Avg_Reward:', round(avg_reward, 2))


if __name__ == "__main__":
    run()

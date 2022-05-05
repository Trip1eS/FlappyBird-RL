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
        self.out = nn.Linear(256, N_ACTIONS)

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.out(x)
        print("out: ", x)
        x = F.softmax(x, dim=1)

        return x


class PolicyGradient(object):
    def __init__(self):
        self.LR = 1e-4  # learning rate
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
        act_prob = self.net.forward(obs).detach().cpu().view(-1).numpy()
        action = np.random.choice(self.N_ACTIONS, p=act_prob)
        print("prob: ", act_prob)
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

        act_prob = self.net.forward(state_pool_tensor)
        print("prob: ", act_prob)
        log_prob = F.cross_entropy(act_prob, action_pool_tensor)
        loss = log_prob * discounted_reward
        loss = torch.mean(loss)
        print("loss: ", loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []


def process_image(image_data):
    x_t1 = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    return x_t1


def process_state(obs, state):
    state = process_image(state)
    state = np.reshape(state, (80, 80, 1))
    state = np.transpose(state, (2, 0, 1))
    cur_obs = copy.deepcopy(obs)
    next_obs = copy.deepcopy(state)
    next_obs = np.append(next_obs, cur_obs[:3, :, :], axis=0)
    return cur_obs, next_obs


bird = PolicyGradient()
env = BirdEnv.BirdEnv()

N_EPISODE = 1000
STEP = 300
TEST = 5


def run_episode():
    s = env.reset()
    img = process_image(s[0])
    obs = np.stack((img, img, img, img), axis=2)
    obs = np.transpose(obs, (2, 0, 1))
    while True:
        env.render()
        action = bird.choose_action(obs)
        state, reward, over, _ = env.step(action)
        cur_obs, next_obs = process_state(obs, state)
        bird.store_transition(cur_obs, action, reward)
        obs = next_obs
        if over:
            break


def evaluate():
    eval_reward = []
    for i in range(TEST):
        episode_reward = 0
        s = env.reset()
        img = process_image(s[0])
        obs = np.stack((img, img, img, img), axis=2)
        obs = np.transpose(obs, (2, 0, 1))
        while True:
            env.render()
            action = bird.choose_action(obs)
            state, reward, over, _ = env.step(action)
            cur_obs, next_obs = process_state(obs, state)
            obs = next_obs
            episode_reward = reward
            if over:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def run():
    for i_episode in range(1000):
        run_episode()
        if i_episode % 5 == 0:
            print("Episode: {}, Reward Sum: {}".format(i_episode, sum(bird.reward_pool)))
        bird.learn()
        if (i_episode + 1) % 10 == 0:
            total_reward = evaluate()
            print("Test Reward: {}".format(total_reward))


if __name__ == "__main__":
    run()

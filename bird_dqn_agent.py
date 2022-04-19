import torch
import torch.nn as nn
import numpy as np

# Define the network used in both target net and the net for training
class NetDQN(nn.Module):
    def __init__(self, N_ACTIONS):
        super(NetDQN, self).__init__()
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
        actions_value = self.out(x)

        return actions_value


class DQN(object):
    def __init__(self):

        self.BATCH_SIZE = 32  # batch size of sampling process from buffer
        self.LR = 0.01  # learning rate
        self.EPSILON = 0.9  # epsilon used for epsilon greedy approach
        self.GAMMA = 0.9  # discount factor
        self.TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target network updates
        self.MEMORY_CAPACITY = 200  # The capacity of experience replay buffer

        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.N_ACTIONS = 2

        # -----------Define 2 networks (target and training)------#
        self.eval_net, self.target_net = NetDQN(self.N_ACTIONS).to(self.DEVICE), NetDQN(self.N_ACTIONS).to(self.DEVICE)
        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # ----Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = [[] for _ in range(self.MEMORY_CAPACITY)]

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)

        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy

        x = x[np.newaxis, :, :, :]
        x = torch.FloatTensor(x)
        x = x.to(self.DEVICE)
        # input only one sample
        if np.random.uniform() < self.EPSILON:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            actions_value = actions_value.cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()  # get max-value.index
            action = action[0]  # return the argmax index
        else:  # random
            action = np.random.randint(0, 1)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % self.MEMORY_CAPACITY  # if memory_counter > MEMORY_CAPACITY, it will update
        l = list()
        l.append(s)
        l.append(a)
        l.append(r)
        l.append(s_)
        self.memory[index] = l
        self.memory_counter += 1

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % self.TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        # extract vectors or matrices s, a, r, s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = list()
        b_a = list()
        b_r = list()
        b_s_ = list()
        for i in sample_index:
            b_s.append(self.memory[i][0])
            b_a.append([self.memory[i][1]])
            b_r.append([self.memory[i][2]])
            b_s_.append(self.memory[i][3])
        b_s = np.array(b_s)
        b_s = torch.FloatTensor(b_s).to(self.DEVICE)
        # convert long int type to tensor
        b_a = torch.LongTensor(b_a).to(self.DEVICE)
        b_r = torch.FloatTensor(b_r).to(self.DEVICE)
        b_s_ = np.array(b_s_)
        b_s_ = torch.FloatTensor(b_s_).to(self.DEVICE)

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

    def save_model(self, path='model.pth'):
        torch.save({
            'eval': self.eval_net.state_dict(),
            'target': self.target_net.state_dict()
        }, path)

    def load_model(self, path='model.pth'):
        file = torch.load(path)
        self.eval_net.load_state_dict(file['eval'])
        self.eval_net.eval()
        self.target_net.load_state_dict(file['target'])
        self.target_net.eval()

    def save_memory(self, path='memory'):
        np.save(path, self.memory)

    def load_memory(self, path='memory'):
        with open(path, 'rb') as f:
            self.memory = np.load(f)
            self.memory_counter = self.MEMORY_CAPACITY + 1

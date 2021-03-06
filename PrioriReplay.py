import numpy as np
import random
from collections import namedtuple,deque

from model import QNetwork
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from operator import itemgetter

Buffer_size = int(1e5)
Batch_size = 64
Gamma = 0.99
Tau = 1e-3
LR = 5e-4
Update_every = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DQNAgent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = PrioriReplayBuffer(action_size, Buffer_size, Batch_size, seed)
        self.t_step = 0
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = self.t_step + 1
        self.t_step = self.t_step % Update_every
        if self.t_step == 0:
            if len(self.memory)>Batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, Gamma)
    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, Gamma):
        states, actions, rewards, next_states, dones, importances, indices = experiences
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (Gamma*Q_target_next*(1-dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        errors = abs(Q_targets-Q_expected)
        loss = F.mse_loss(Q_targets, Q_expected)
        errors = abs(Q_targets.detach().numpy()-Q_expected.detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        for param, weight in zip(self.qnetwork_local.parameters(), importances):
            weight = np.float64(weight)
            param.data *= weight
        self.optimizer.step()
        self.memory.set_priorities(indices, errors)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, Tau)
    
    def soft_update(self, local_model, target_model, Tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(Tau*local_param.data + (1.0-Tau)*target_param.data)
            

class PrioriReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = []
        self.priorities = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.priorities.append(max(self.priorities, default = 1))
        self.memory.append(e)
        
        
    def get_probs(self, a = 0):
        scaled_priorities = np.array(self.priorities)**a
        sample_probs = scaled_priorities/sum(scaled_priorities)
        return sample_probs
    
    def get_importance(self, probabilities):
        importance = (1/len(self.memory))*(1/probabilities)
        importance_normalized = importance/max(importance)
        return importance_normalized
    def set_priorities(self, indices, errors, offset = 0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = e + offset
    
    def sample(self):
        sample_probs = self.get_probs()
        sample_indices = random.choices(range(len(self.memory)), k = Batch_size, weights = sample_probs)
        experiences = itemgetter(*sample_indices)(self.memory)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        importances = self.get_importance(sample_probs[sample_indices])
        return(states, actions, rewards, next_states, dones, importances, sample_indices)
    def write_file(self):
        
        f = open("dict.json","w")
        pickle.dump(random.sample(self.memory, k = 500), f)
        
    def __len__(self):
        return len(self.memory)
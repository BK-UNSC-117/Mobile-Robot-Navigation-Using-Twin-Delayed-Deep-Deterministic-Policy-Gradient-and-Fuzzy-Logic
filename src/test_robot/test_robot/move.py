from env import turtle_sim_env
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import rclpy

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims,
            n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.device = T.device('cuda')

        self.to(self.device)

    def forward(self, state):
        state = T.FloatTensor(state).to(self.device)
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu

def main(args=None):
    rclpy.init(args=args)
    env=turtle_sim_env()
    actor=ActorNetwork(3,400,300,2)
    actor.load_state_dict(T.load("/home/doraemon/mobot_navi/src/training/training/actor td3"))

    rclpy.spin_once(env)
    distance=[]
    while True:
        done=False
        state,_=env.reset()
        distance.append(state[0])
        while not done:
            act=actor(state)
            next_state,done=env.step(act)
            state=next_state
            distance.append(state[0])
        np.savetxt("/home/doraemon/mobot_navi/src/test_robot/test_robot/distances.txt",np.array(distance))

main()


from env import DRLEnvironment
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import rclpy
from fuzzy_ctrl import fuz_ctrl


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
    env=DRLEnvironment()
    fuz=fuz_ctrl()
    actor=ActorNetwork(3,400,300,2)
    actor.load_state_dict(T.load("/home/doraemon/mobot_navi/src/training/training/actor td3"))

    rclpy.spin_once(env)
    dist=[]
    while True:
        done=False
        state,scan_range,_=env.reset()
        while np.array(scan_range).shape[0]==0:
            rclpy.spin_once(env)
            state,scan_range,_=env.reset()
        min_range=np.min(scan_range)
        
        degree=np.argmin(scan_range)+1
        dist.append(state[0])
        while not done:
            if min_range<0.07:
                print("min range: "+str(min_range)+" at degree: "+str(degree))
                lin,ang=fuz.get_action(degree,min_range)
                action=[lin,ang]
                state,scan_range,done=env.step(action)
                min_range=np.min(scan_range)
                degree=np.argmin(scan_range)+1
                dist.append(state[0])

            else:
                action=actor(state)
                print("dist: "+str(state[0])+" heading: "+str(state[1])+\
                        " min range: "+str(min_range)+" degree: "+str(degree))
                state,scan_range,done=env.step(action)
                min_range=np.min(scan_range)
                degree=np.argmin(scan_range)+1
                dist.append(state[0])
            np.savetxt("/home/doraemon/mobot_navi/src/fuzzy_rl/fuzzy_rl/distances.txt",np.array(dist))
main()


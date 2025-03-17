#!/usr/lin/env python3

import random
import matplotlib.pyplot as plt
import time
import numpy as np
import torch

import rclpy.executors

import rclpy
import threading
from env import turtle_sim_env

from td3 import Agent

p_init=0.7
p_end=0.1
r=1e-6

def main(args=None):
    rclpy.init(args=args)

    env=turtle_sim_env()
    exe=rclpy.executors.MultiThreadedExecutor(8)
    exe.add_node(env)
    exe_thrd=threading.Thread(target=exe.spin,daemon=True)
    exe_thrd.start()

    n_action=2
    rclpy.spin_once(env)
    state,done=env.reset()
    
    n_state=len(state)
    print(n_state)
    print(n_action)
    agnt=Agent(alpha=0.001,beta=0.001,input_dims=n_state,tau=0.005,
           batch_size=100,layer1_size=400,layer2_size=300,
           n_actions=2)
    
    scores,losses_act,q1_losses,q2_losses=[],[],[],[]

    #scores=list(np.loadtxt(r"/home/doraemon/mobot_navi/src/training/training/scores.txt"))
    #losses_act=list(np.loadtxt(r"/home/doraemon/mobot_navi/src/training/training/actor_loss.txt"))
    #q1_losses=list(np.loadtxt(r"/home/doraemon/mobot_navi/src/training/training/q1_loss.txt"))
    #q2_losses=list(np.loadtxt(r"/home/doraemon/mobot_navi/src/training/training/q2_loss.txt"))
    
    n_iter=100

    for i in range(n_iter):
        score=0
        actor_loss=0
        q1_loss=0
        q2_loss=0
        done=False
        state,done=env.reset()
        counter=0
        while not done:
            counter+=1
            act=agnt.choose_action(state)
            next_state,reward,done=env.step(act)
            score+=reward
            agnt.remember(state,act,reward,next_state,done)
        
            agnt.learn()
            state=next_state

            actor_loss+=agnt.loss_actor
            q1_loss+=agnt.loss_q1
            q2_loss+=agnt.loss_q2
        
        print("iteration: "+str(i+66)+" score: "+str(score)+" number of moves: "+str(counter)+" epsilon: "+str(agnt.eps))
        scores.append(score)
        losses_act.append((actor_loss.item())/counter)
        q1_losses.append((q1_loss.item())/counter)
        q2_losses.append((q2_loss.item())/counter)
        
        np.savetxt("scores.txt",np.array(scores))
        np.savetxt("actor_loss.txt",losses_act)
        np.savetxt("q1_loss.txt",q1_losses)
        np.savetxt("q2_loss.txt",q2_losses)

if __name__=='__main__':
    main()

#!/usr/lin/env python3

import numpy as np
import time
import math

import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from turtlesim.srv import TeleportAbsolute

class turtle_sim_env(Node):
    def __init__(self):
        super().__init__("turtle_sim_env")
        self.cmd_pub=self.create_publisher(Twist,
            "/turtle1/cmd_vel",10)
        self.pse_sub=self.create_subscription(Pose,
            "/turtle1/pose",self.get_state,10)
        
        self.reset_proxy=self.create_client(Empty,"reset")
        self.teleport=self.create_client(TeleportAbsolute,'/turtle1/teleport_absolute')
        self.telep=TeleportAbsolute.Request()
        
        self.robot_x_init=np.random.uniform(low=2,high=4,size=1)[0]
        self.robot_y_init=np.random.uniform(low=4,high=6,size=1)[0]

        self.goal_x=np.random.uniform(low=6.5,high=8,size=1)[0]
        self.goal_y=np.random.uniform(low=2,high=8,size=1)[0]
        print("goal: "+str([self.goal_x,self.goal_y]))

        self.telep.x = self.robot_x_init
        self.telep.y = self.robot_y_init
        self.telep.theta = 0.0
        self.teleport.call_async(self.telep)

        self.done=False
        self.get_goalbox=False
        self.screwed=False
        self.init=True

        self.ang=0
        self.ang_n=0
        self.lin=0
        self.lin_n=0

        self.start_time=time.time()

        self.dist_init=0

        self.goal_dist=0

        self.action=[0,0]
        self.state=[]
        self.state_init=[]

        self.prize=0

    def get_state(self,msg):
        if self.init:
            self.dist_init=math.sqrt((self.robot_x_init-self.goal_x)**2+(self.robot_y_init-self.goal_y)**2)
            self.heading_init=math.atan2((-self.robot_y_init+self.goal_y),(-self.robot_x_init+self.goal_x))
            while self.heading_init>np.pi:
                self.heading_init-=2*np.pi
            while self.heading_init<np.pi:
                self.heading_init+=2*np.pi
            self.init=False
            print("init mode finished")
            self.heading=self.heading_init
            self.goal_dist=self.dist_init
            self.current_time=0
            self.state_init=np.array([self.dist_init/(self.dist_init+1),
                                  self.heading_init/np.pi,
                                  0])
            self.state=self.state_init
        else:
            self.robot_angle=msg.theta
            self.robot_x=msg.x
            self.robot_y=msg.y

            self.diff_x=self.goal_x-self.robot_x
            self.diff_y=self.goal_y-self.robot_y
        
            self.goal_dist=math.sqrt(self.diff_x**2+self.diff_y**2)
            self.goal_angle=math.atan2(self.diff_y,self.diff_x)
            self.heading=self.goal_angle-self.robot_angle
            

            self.current_time=time.time()-self.start_time

            if self.goal_dist<0.3:
                self.reset_proxy.call_async(Empty.Request())
                self.get_goalbox=True
                print("reached the goal :)))))))))))))))))")
                self.change_goal()

            if self.goal_dist>self.dist_init+1:
                self.screwed=True
                self.init=True
                print("screwed")

            while self.heading>np.pi:
                self.heading-=2*np.pi
            while self.heading<-np.pi:
                self.heading+=2*np.pi

            self.state=np.array([self.goal_dist/(self.dist_init+1),
                             self.heading/np.pi,
                             0])
    
    def reward(self):
        '''
        r_yaw=-1*abs(self.heading)
        r_dist=(2*self.dist_init)/(self.dist_init+self.goal_dist)-1
        r_ang=-1*(self.ang**2)
        r_lin=-1*(((0.22-self.lin)*10)**2)
        if self.screwed:
            done=True
            reward=-2000
        elif self.get_goalbox or self.current_time>60:
            reward=r_yaw+r_dist+r_ang+r_lin-1
            done=True
        else:
            done=False
            reward=r_yaw+r_dist+r_ang+r_lin-1
        '''
        if self.screwed:
            done=True
            reward=-1000
        elif self.get_goalbox or self.current_time>60:
            reward=-(self.goal_dist-self.dist_init)-abs(self.heading)*self.goal_dist
            done=True
        else:
            done=False
            reward=-(self.goal_dist-self.dist_init)-abs(self.heading)*self.goal_dist
        return reward,done
    
    def step(self,action):
        cmd=Twist()
        self.ang_n=self.action[0]
        self.lin_n=self.action[1]
        self.action[0]=(action[0]+1)*0.22
        self.action[1]=action[1]*2.28
        self.lin=self.action[0]
        self.ang=self.action[1]
        cmd.linear.x=float(self.action[0])
        cmd.angular.z=float(self.action[1])
        self.cmd_pub.publish(cmd)

        rclpy.spin_once(self)

        self.prize,self.done=self.reward()

        return self.state,self.prize,self.done

    def reset(self):
        self.reset_proxy.call_async(Empty.Request())
        self.done=False
        self.screwed=False
        self.get_goalbox=False
        self.init=True
        self.rotate=[]
        self.reward_sum=0
        reset_cmd=Twist()
        reset_cmd.linear.x=0.0
        reset_cmd.angular.z=0.0
        self.cmd_pub.publish(reset_cmd)

        print("waiting...")
        time.sleep(2)

        self.telep.x = self.robot_x_init
        self.telep.y = self.robot_y_init
        self.telep.theta = 0.0
        self.teleport.call_async(self.telep)

        self.start_time=time.time()

        return self.state_init,self.done
    
    def change_goal(self):
        self.robot_x_init=np.random.uniform(low=2,high=4,size=1)[0]
        self.robot_y_init=np.random.uniform(low=4,high=6,size=1)[0]

        self.goal_x=np.random.uniform(low=6.5,high=8,size=1)[0]
        self.goal_y=np.random.uniform(low=2,high=8,size=1)[0]
        print("new goal: "+str([self.goal_x,self.goal_y]))

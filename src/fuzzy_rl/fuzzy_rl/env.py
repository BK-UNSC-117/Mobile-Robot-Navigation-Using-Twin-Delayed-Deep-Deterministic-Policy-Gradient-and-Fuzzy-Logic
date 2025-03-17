#!/usr/lin/env python3

import random
import math
import numpy as np
from numpy.core.numeric import Infinity
import time

from geometry_msgs.msg import Pose, Twist
import rclpy.executors
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

goal_points=np.array([[-0.08,0.54],[0.67,-1.27],[1.08,0.66],[2.27,0.56],
                      [0.2,0.6],[0.2,0.6],[1.63,1.76],[1.65,-1.75]])

NUM_SCAN_SAMPLES=360
LINEAR = 0
ANGULAR = 1
MAX_GOAL_DISTANCE=math.sqrt(4.2**2+4.2**2)
pi=np.pi

class DRLEnvironment(Node):

    def __init__(self):
        super().__init__('drl_environment')
        self.range=[]
        self.episode_timeout=50
        self.goal_position=Pose()
        self.scan_topic='scan'
        self.velo_topic='cmd_vel'
        self.odom_topic='odom'
        self.goal_topic='goal_pose'

        self.robot_x_init=-1.9999421375393494
        self.robot_y_init=-0.5000009930209561

        self.lin_o=0
        self.ang_o=0


        self.action=[0.0,0.0]

        self.goal_x,self.goal_y=2.27,0.56
        self.robot_x,self.robot_y=self.robot_x_init,self.robot_y_init

        self.goal_dist_init=0

        self.upper=5.0
        self.lower=-5.0

        self.action_linear_previous=0
        self.action_angular_previous=0
        self.robot_x_prev,self.robot_y_prev=0.0,0.0
        self.robot_heading=0.0
        self.total_distance=0.0
        self.robot_tilt=0.0

        self.done=False
        self.succeed=0
        self.episode_deadline=Infinity
        self.reset_deadline=False
        self.get_goalbox=False
        self.clock_msgs_skipped=0

        self.new_goal=False
        self.goal_angle=0.0
        self.goal_distance=MAX_GOAL_DISTANCE
        self.initial_distance_to_goal=MAX_GOAL_DISTANCE

        self.start_time=time.time()
        self.spent=0

        self.heading_init=0

        self.obstacle_distance=3.5

        self.scan_range=[]
        self.difficulty_radius=1
        self.local_step=0
        self.time_sec=0

        self.odom_received = False

        qos = QoSProfile(depth=10)
        qos_clock = QoSProfile(depth=1)
        self.cmd_vel_pub = self.create_publisher(Twist,self.velo_topic,qos)
        self.odom_sub = self.create_subscription(Odometry,self.odom_topic,
                                                 self.odom_callback,qos)
        self.scan_sub = self.create_subscription(LaserScan,'scan', 
                    self.getState,qos_profile_sensor_data)
        self.unpause=self.create_client(Empty,"/unpause_physics")
        self.pause=self.create_client(Empty,"/pause_physics")
        self.reset_proxy=self.create_client(Empty,"/reset_world")

    def odom_callback(self, msg):

        self.odom_received = True
        self.goal_dist_init=math.sqrt((self.robot_x_init-self.goal_x)**2+(self.robot_y_init-self.goal_y)**2)+1
        self.heading_init=math.atan2((-self.robot_x_init+self.goal_x),(-self.robot_y_init+self.goal_y))
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _,_,self.robot_heading=euler_from_quaternion([q.x,q.y,
                                                    q.z,q.w])
        self.robot_tilt = msg.pose.pose.orientation.y

        diff_y=self.goal_y-self.robot_y
        diff_x=self.goal_x-self.robot_x
        distance_to_goal=math.sqrt(diff_x**2 + diff_y**2)
        heading_to_goal=math.atan2(diff_y, diff_x)
        goal_angle=heading_to_goal-self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi
        

        self.goal_distance=distance_to_goal
        self.goal_angle=goal_angle
        self.heading=heading_to_goal

        self.state=np.array([self.goal_distance/(self.goal_dist_init+1),
                                                self.goal_angle/np.pi,
                                                                    0])
    
    def getState(self,scan):
        while not self.odom_received:
            rclpy.spin_once(self)
            print(".....")
        self.done=False
        min_range = 0.12
        self.scan_range=np.array(scan.ranges)
        self.scan_range[self.scan_range==np.inf]=3.5
        self.scan_range[self.scan_range<0.12]=0.12

        obstacle_min_range = round(min(self.scan_range), 2)

        self.spent=(time.time()-self.start_time)
        
        if self.goal_distance < 0.2:
            self.get_goalbox = True
            print("Reached the goal")
        
        self.scan_range=np.array(self.scan_range)/3.5
    
    def step(self,act):
        cmd=Twist()
        self.action[0]=act[0]
        self.action[1]=act[1]
        cmd.linear.x=float((self.action[0]+1)*0.5*0.22)
        cmd.angular.z=float(self.action[1]*2.24)
        self.cmd_vel_pub.publish(cmd)

        rclpy.spin_once(self)
        while np.array(self.scan_range).shape[0]==0:
            rclpy.spin_once(self)
            print("connecting....")

        self.action_linear_previous=self.action[0]
        self.action_angular_previous=self.action[1]
        return np.array(self.state),np.array(self.scan_range),self.done or self.get_goalbox
    
    def reset(self):
        while not self.odom_received:
            rclpy.spin_once(self)
            print(".....")
        self.done=False
        self.get_goalbox=False
        self.cmd_vel_pub.publish(Twist())
        self.reset_proxy.call_async(Empty.Request())
        self.unpause.call_async(Empty.Request())

        time.sleep(3)

        rclpy.spin_once(self)
        while np.all(self.scan_range[:NUM_SCAN_SAMPLES] == 0):
            rclpy.spin_once(self)
        
        self.start_time=time.time()

        return self.state,self.scan_range,self.done
    
    def change_goal(self):
        goal_point=random.choice(goal_points)
        self.goal_x=goal_point[0]
        self.goal_y=goal_point[1]
        print("new_goal: "+str([self.goal_x,self.goal_y]))

def main(args=None):
    rclpy.init(args=args)

    env=DRLEnvironment()

    state,done=env.reset()
    state=np.array(env.range)
    while state.shape[0]==0:
        rclpy.spin_once(env)
        state,done=env.reset()
        state=np.array(state)
        print("started")
    rclpy.spin_once(env)
    state,done=env.reset()
    print("state: "+str(state))
    time.sleep(2)
    
    while not done:
        rclpy.spin_once(env)
        next_state,reward,done=env.step([float(np.random.uniform(low=0.0,high=0.22,size=1)),
                                float(np.random.uniform(low=-2.84,high=2.84,size=1))])
        
        print("next_state: "+str(next_state))
        

#if __name__=='__main__':
#    main()

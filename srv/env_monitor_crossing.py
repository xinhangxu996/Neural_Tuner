import rospy
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

from traj_planner.msg import Weights
from people_msgs.msg import People
from nav_msgs.msg import Odometry
from neural_tuner.msg import Observation
import threading
import random
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
import math
from tf.transformations import quaternion_from_euler
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import tf
from geometry_msgs.msg import PoseStamped

class EnvMonitor:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('Env_monitor', anonymous=True)
        
        #map setting
        self.localmap_shape=np.array((50,50))         #map size 30 width 30 length
        self.resolution=0.2                           #resolution of local map 0.2 for x and y axis
        self.localmap_center_index=self.localmap_shape/2 # the map center is the viechles
        self.extend_scale=5
        #vel map setting
        self.velocitymap_shape=self.localmap_shape*1  #size of vel map is 10 times of localmap
        self.velocity_integrate_time=0.5              #velocity 0.5 seconds distance
        self.people_color =1
        self.vel_color=2
        self.vel_width=1

        #State related variable
        self.current_viechle_pose= None
        self.current_viechle_yaw = None
        self.current_viechle_vel = None

        self.current_people_map=None
        self.current_obs_map=None

        self.current_parameters=None

        self.planning_success=None
        self.collision=None
        self.tracking_error=None
        self.reached=None

        #ENV control flag and logic variable

        self.sent_command=False
        self.reset_request=False
        
        #desired variable
        self.goal=None
        self.goal_yaw=None
        self.angle_bias=30
        #reward
        self.reward=0
        self.need_reset=1

        self.cost_collision=20
        self.cost_planning_failed=1500
        self.reward_reach=50
        self.cost_step=0.2
        self.award_plan_suc=0.1
        self.scale=500

        #reset check
        self.failed_time=0
        self.last_failed_time=0
        self.last_successful_time=0


        # Create publisher for the image test and reset command

        self.image_publisher = rospy.Publisher('/obst_map_image', Image, queue_size=10)
        self.image_publisher_1 = rospy.Publisher('/people_image', Image, queue_size=10)
        self.observation_publisher = rospy.Publisher('/obs_state', Observation, queue_size=1)

        self.people_pub_1=rospy.Publisher('/actor1/reset_command', Float32MultiArray, queue_size=1)
        self.people_pub_2=rospy.Publisher('/actor2/reset_command', Float32MultiArray, queue_size=1)
        self.people_pub_3=rospy.Publisher('/actor3/reset_command', Float32MultiArray, queue_size=1)
        self.people_pub_4=rospy.Publisher('/actor4/reset_command', Float32MultiArray, queue_size=1)
        self.people_pub_5=rospy.Publisher('/actor5/reset_command', Float32MultiArray, queue_size=1)
        self.people_pub_6=rospy.Publisher('/actor6/reset_command', Float32MultiArray, queue_size=1)
        
        self.planner_pub=rospy.Publisher('/reset_planner', Int32, queue_size=1)
        self.goal_pub=rospy.Publisher('/shortterm_goal',PoseStamped, queue_size=1)
        self.weights_pub=rospy.Publisher('/set_weights',Weights,queue_size=1)
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        self.map_sub=rospy.Subscriber('/obst_map', Int32MultiArray, self.obst_map_cb) 
        
        self.status_sub=rospy.Subscriber('/using_weights',Weights,self.status_cb)
        
        self.people_sub=rospy.Subscriber('/people',People,self.people_cb)

        self.odom_sub=rospy.Subscriber('/odom',Odometry,self.odom_cb)

        self.reset_sub=rospy.Subscriber('/reset_command',Int32,self.reset_cb)

        self.run_sub=rospy.Subscriber('/env_run',Int32,self.run_cb)

        self.action_sub = rospy.Subscriber('/action_array', Float32MultiArray, self.action_callback)

        self.steps=0

        # Set the timer to call the callback function every 1 second (1.0 seconds)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        
        rospy.spin()
    
    def obst_map_cb(self,data):
        # Convert the 1D list to a 30x30 numpy array
        matrix = np.array(data.data).reshape(self.localmap_shape[0], self. localmap_shape[1])
        # Flip both vertically and horizontally
        matrix = np.flip(matrix, axis=(0,1))
        self.current_obs_map=matrix
        # Normalize the matrix to the range [0, 255] for display
        normalized_matrix = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to an 8-bit grayscale image
        image = np.uint8(normalized_matrix)
        # Convert the image to a ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
        # Publish the image
        self.image_publisher.publish(ros_image)
        # rospy.loginfo("Published obstacle map image.")

    def status_cb(self,data):
        self.reached=data.reached
        self.current_parameters=np.array((data.wei_obs,data.wei_surround,data.wei_feas,data.wei_sqrvar,data.wei_time,data.wei_jerk))
        self.planning_success=data.planning_success
        self.collision=data.collision
        self.tracking_error=data.tracking_error
        self.reward_calculate()
        self.reset_check()

    def people_cb(self,data):

        people_pic_large=np.zeros((self.localmap_shape[0]*self.extend_scale,self.localmap_shape[1]*self.extend_scale))

        if self.current_viechle_pose is None:
            return
        pos_viechle_current=self.current_viechle_pose

        people_pos_index=[]

        for item in data.people:
            pos_vec=np.array((item.position.x,item.position.y))
            true_relative_pos=pos_vec-pos_viechle_current
            pos_relative=pos_vec-pos_viechle_current+self.resolution*self.localmap_shape*self.extend_scale/2

            if np.linalg.norm(pos_vec-pos_viechle_current)>self.localmap_shape[0]/2*self.resolution:
                continue
            
            if abs(true_relative_pos[0])>self.localmap_shape[0]/2*self.resolution or abs(true_relative_pos[1])>self.localmap_shape[0]/2*self.resolution:
                continue 

            vel_vec=np.array((item.velocity.x,item.velocity.y))
            vel_norm=np.linalg.norm(vel_vec)
            
            if vel_norm>12:
                vel_vec=vel_vec*12/vel_norm
            
            pos_after_integrate=pos_relative+vel_vec*self.velocity_integrate_time

            pos_index=pos_relative//self.resolution
            pos_end_index=pos_after_integrate//self.resolution
            pt1 = (int(pos_index[1]), int(pos_index[0]))
            pt2 = (int(pos_end_index[1]), int(pos_end_index[0]))
            cv2.line(people_pic_large, pt1, pt2, self.vel_color, self.vel_width)            
            people_pos_index.append(pos_index)

        for item in people_pos_index:
            people_pic_large[int(item[0]), int(item[1])] = self.people_color        
        #Trim the image
        Trim_low=self.localmap_shape*(self.extend_scale//2)
        Trim_high=self.localmap_shape*(self.extend_scale//2+1)
        people_pic=people_pic_large[Trim_low[0]:Trim_high[0],Trim_low[1]:Trim_high[1]]

        self.current_people_map=people_pic

        # Normalize the matrix to the range [0, 255] for display
        normalized_matrix = cv2.normalize(people_pic, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to an 8-bit grayscale image
        image = np.uint8(normalized_matrix)
        # Convert the image to a ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
        # Publish the image
        self.image_publisher_1.publish(ros_image)
        # rospy.loginfo("Published obstacle map image.")
    
    def odom_cb(self,data):
        # Extract position
        position = data.pose.pose.position
        x = position.x
        y = position.y
        z = position.z

        # Extract velocity
        velocity = data.twist.twist.linear
        vx = velocity.x
        vy = velocity.y
        vz = velocity.z

        # Extract pose quaternion
        orientation = data.pose.pose.orientation
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w

        # Convert quaternion to Euler angles
        euler = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]  # This is the heading (yaw) angle in radians

        self.current_viechle_pose=np.array((x,y))
        self.current_viechle_vel =np.array((vx,vy))
        self.current_viechle_yaw =yaw
    
    def timer_callback(self, event):
        if self.need_reset :
            rospy.loginfo("waiting reset")
            return
        
        if not self.sent_command:
            rospy.loginfo("set command return")
            return
        
        if self.reset_request:
            rospy.loginfo("request return")
            return
        
        if self.current_viechle_pose is None or self.current_viechle_yaw is None or self.current_viechle_vel is None or self.current_people_map is None or self.current_obs_map is None or self.current_parameters is None or self.planning_success is None or self.collision is None or self.tracking_error is None :
            rospy.loginfo("others return")
            return
        
        #Genrate the new msg
        msg=Observation()
        msg.map_size=self.localmap_shape.flatten().tolist()
        msg.current_viechle_pose=self.current_viechle_pose.flatten().tolist()
        msg.current_viechle_yaw=self.current_viechle_yaw
        msg.current_viechle_vel=self.current_viechle_vel.flatten().tolist()
        msg.current_people_map=self.current_people_map.flatten().tolist()
        msg.current_obs_map=self.current_obs_map.flatten().tolist()
        msg.current_parameters=self.current_parameters.flatten().tolist()
        msg.planning_success=self.planning_success
        msg.collision=self.collision
        msg.tracking_error=self.tracking_error
        msg.reward=self.reward
        msg.need_reset=self.need_reset


        self.observation_publisher.publish(msg)
        self.reward=0

    def reward_calculate(self):
        self.reward=self.reward-self.cost_collision*self.collision
        self.reward=self.reward-self.cost_step
        if self.planning_success==0:
            self.reward=self.reward+self.award_plan_suc
        return  
    def action_callback(self,msg):

        new_weight_msg=Weights()
        
        weight_list=list(msg.data)

        # new_weight_msg.wei_obs=weight_list[0]
        # new_weight_msg.wei_surround=weight_list[1]
        # new_weight_msg.wei_feas=weight_list[2]
        # new_weight_msg.wei_sqrvar=weight_list[3]
        # new_weight_msg.wei_time=weight_list[4]

        new_weight_msg.wei_obs=abs(weight_list[0]*self.scale)
        new_weight_msg.wei_surround=abs(weight_list[1]*self.scale)
        new_weight_msg.wei_feas=abs(weight_list[2]*self.scale)
        new_weight_msg.wei_sqrvar=abs(weight_list[3]*self.scale)
        new_weight_msg.wei_time=abs(weight_list[4]*self.scale)
        new_weight_msg.wei_jerk=abs(weight_list[5]*self.scale)
        self.weights_pub.publish(new_weight_msg)

    def reset_cb(self,data):
        self.reset()
        return
    def run_cb(self,data):
        # while(self.need_reset):
        #     do_nothing=1
        print(self.need_reset)
        # Create a PoseStamped message
        pose = PoseStamped()
        quaternion=quaternion_from_euler(0, 0, self.goal_yaw)
        # Set the header (for time and frame ID)
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"  # Change this to the relevant coordinate frame

        # Set position (in meters)
        pose.pose.position.x = self.goal[0]
        pose.pose.position.y = self.goal[1]
        pose.pose.position.z = 0.0

        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        # # Publish the PoseStamped message
        self.goal_pub.publish(pose)

        # rospy.loginfo("Published Pose: %s" % pose)
        self.last_successful_time=rospy.Time.now().to_sec()
        self.sent_command=True
        self.reward=0

        return  

    def reset(self):
        #State related variable
        self.reached=None
        self.current_viechle_pose= None
        self.current_viechle_yaw = None
        self.current_viechle_vel = None
        self.current_people_map=None
        self.current_obs_map=None
        self.current_parameters=None
        self.planning_success=None
        self.collision=None
        self.tracking_error=None
        #ENV control flag and logic variable
        self.sent_command=False
        self.reset_request=False
        #reward
        self.planner_reset()
        Time_current=rospy.Time.now().to_sec()
        self.gazebo_reset()
        self.reward=0
        self.need_reset=0
        return
    def gazebo_reset(self):

        people_pos=self.generate_arrays_crossing(6)
        
        car_pos=self.generate_car()
        
        # Define the message content
        people_msg = Float32MultiArray()
        people_msg.data = people_pos[0,:].flatten().tolist()
        self.people_pub_1.publish(people_msg)
        people_msg.data = people_pos[1,:].flatten().tolist()
        self.people_pub_2.publish(people_msg)
        people_msg.data = people_pos[2,:].flatten().tolist()
        self.people_pub_3.publish(people_msg)
        people_msg.data = people_pos[3,:].flatten().tolist()
        self.people_pub_4.publish(people_msg)
        people_msg.data = people_pos[4,:].flatten().tolist()
        self.people_pub_5.publish(people_msg)
        people_msg.data = people_pos[5,:].flatten().tolist()
        self.people_pub_6.publish(people_msg)

        self.goal=np.array([car_pos[2],car_pos[3]])
        self.set_model_position(car_pos[0],car_pos[1],0.5,car_pos[4])
        
    def planner_reset(self):
        msg=Int32()
        self.planner_pub.publish(msg)
        return

    def valid_combination(self,a1, a2, a3):
        return (
            len(set(a1) & set(a2)) <= 1 and
            len(set(a1) & set(a3)) <= 1 and
            len(set(a2) & set(a3)) <= 1
        )

    def generate_arrays_crossing(self,number):
        group1=[0,1,2,3,4,5]
        group2=[6,7,8,9,10,11]
        random.shuffle(group1)
        random.shuffle(group2)
        array1 = [group1[0], group2[0]]
        array2 = [group1[1], group2[1]]
        array3 = [group1[2], group2[2]]
        array4 = [group1[3], group2[3]]
        array5 = [group1[4], group2[4]]
        array6 = [group1[5], group2[5]]
        random.shuffle(array1)
        random.shuffle(array2)
        random.shuffle(array3)
        random.shuffle(array4)
        random.shuffle(array5)
        random.shuffle(array6)
        
        point_list=[array1,array2,array3,array4,array5,array6]

        point_set = np.array([[12,1.5,1],[12, -1.5 ,1],[1.5, 12 ,1],[-1.5, 12, 1],[6,0,1],[0,6,1],[-12,1.5,1],[-12, -1.5 ,1],[1.5, -12 ,1],[-1.5, -12, 1],[-6,0,1],[0,-6,1]])
        
        random_numbers = np.random.randint(2, size=6)
        
        result=np.zeros((6,4))

        for i in range(number):
            index=int(random_numbers[i])
            result[i,:]=np.array([point_set[point_list[i][index]][0],point_set[point_list[i][index]][1],point_set[point_list[i][1-index]][0],point_set[point_list[i][1-index]][1]])
        
        for i in range(number,3):
            result[i,:]=np.array([100,100,200,200])*i

        return result


    def generate_arrays(self,number):
        group1 = [0, 1, 2]  
        group2 = [3, 4, 5]  
        random.shuffle(group1)
        random.shuffle(group2)

        array1 = [group1[0], group2[0]]
        array2 = [group1[1], group2[1]]
        array3 = [group1[2], group2[2]]

        random.shuffle(array1)
        random.shuffle(array2)
        random.shuffle(array3)

        while not self.valid_combination(array1, array2, array3):
            random.shuffle(group1)
            random.shuffle(group2)
            array1 = [group1[0], group2[0]]
            array2 = [group1[1], group2[1]]
            array3 = [group1[2], group2[2]]
            random.shuffle(array1)
            random.shuffle(array2)
            random.shuffle(array3)
        point_list=[array1,array2,array3]
        point_set = np.array([[-3.55, -2.55, 1.25],[3, 0 ,1.25],[-3.50, -4.10, 1.25],[-3.00, 0.00, 1.25],[2.74, -2.50, 1.25],[2.66, -4.00, 1.25]])
        random_numbers = np.random.randint(2, size=3)
        result=np.zeros((3,4))
        for i in range(number):
            index=int(random_numbers[i])
            result[i,:]=np.array([point_set[point_list[i][index]][0],point_set[point_list[i][index]][1],point_set[point_list[i][1-index]][0],point_set[point_list[i][1-index]][1]])
        for i in range(number,3):
            result[i,:]=np.array([100,100,200,200])*i
        return result

    def generate_car(self):
        point_set=np.array([[9,0],[0,9],[0,-9],[-9,0]])
        
        possible_case=np.array([[0,1],[0,2],[0,3],[1,0],[1,2],[1,3],[2,0],[2,1],[2,3],[3,0],[3,1],[3,2]])
        # Randomly generate an integer between 0 and 7 (inclusive)
        
        random_number = np.random.randint(0, 12)
        
        select_case=possible_case[random_number,:]
        
        start_point=point_set[select_case[0],:]
        
        end_point=point_set[select_case[1],:]

        # Extract the x, y components of the direction vector
        x, y = end_point[0]-start_point[0], end_point[1]-start_point[1]
        
        # Calculate the original angle in the XY plane
        original_angle = math.atan2(y, x)
        self.goal_yaw=original_angle
        # Convert the bias from degrees to radians
        bias_in_radians = math.radians(self.angle_bias)
        
        # Generate a random angle within ±bias of the original angle
        random_angle = original_angle + random.uniform(-bias_in_radians, bias_in_radians)
        
        # Convert the new angle back to a direction vector
        new_x = math.cos(random_angle)
        new_y = math.sin(random_angle)
        
        # Assuming no rotation in the z-axis, generate the corresponding quaternion
        # Use quaternion_from_euler (roll, pitch, yaw)
        yaw = random_angle  # The angle in the XY plane is the yaw

        return np.array([start_point[0],start_point[1],end_point[0],end_point[1],yaw])
    
    def set_model_position(self,x, y, z, yaw):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

            # Convert yaw angle to quaternion (roll and pitch are 0 for flat-plane rotation)
            quaternion = quaternion_from_euler(0, 0, yaw)

            model_state = ModelState()
            model_state.model_name = 'ranger_mini_v2'  # Model name in Gazebo
            model_state.pose.position.x = x
            model_state.pose.position.y = y
            model_state.pose.position.z = z
            
            # Set the orientation using the quaternion
            model_state.pose.orientation.x = quaternion[0]
            model_state.pose.orientation.y = quaternion[1]
            model_state.pose.orientation.z = quaternion[2]
            model_state.pose.orientation.w = quaternion[3]

            # Call the service to set the model's state
            response = set_state_service(model_state)
            if response.success:
                rospy.loginfo("Model position set successfully.")
            else:
                rospy.logwarn("Failed to set model position: " + response.status_message)
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
    def reset_check(self):
        if self.need_reset or not self.sent_command:
            return
        if self.planning_success==0:
            self.last_successful_time=rospy.Time.now().to_sec()
        if self.planning_success>0:
            self.last_failed_time=rospy.Time.now().to_sec()
        if self.last_failed_time - self.last_successful_time > 3.0 or self.reached:
            rospy.loginfo("Reset condition met. Last failed time: %.2f, Last successful time: %.2f, Time difference: %.2f seconds",
                        self.last_failed_time, self.last_successful_time, self.last_failed_time - self.last_successful_time)
            if not self.reached:
                self.reward=self.reward-self.cost_planning_failed
            if self.reached:
                self.reward=self.reward+self.reward_reach
            
            self.need_reset = 1
            #Genrate the new msg
            msg=Observation()
            msg.map_size=self.localmap_shape.flatten().tolist()
            msg.current_viechle_pose=self.current_viechle_pose.flatten().tolist()
            msg.current_viechle_yaw=self.current_viechle_yaw
            msg.current_viechle_vel=self.current_viechle_vel.flatten().tolist()
            msg.current_people_map=self.current_people_map.flatten().tolist()
            msg.current_obs_map=self.current_obs_map.flatten().tolist()
            msg.current_parameters=self.current_parameters.flatten().tolist()
            msg.planning_success=self.planning_success
            msg.collision=self.collision
            msg.tracking_error=self.tracking_error
            msg.reward=self.reward
            msg.need_reset=self.need_reset


            self.observation_publisher.publish(msg)
            self.reward=0

if __name__ == '__main__':
    try:
        # Create an instance of the class and run the monitor method
        monitor_node = EnvMonitor()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from pynput import keyboard

def publish_reset_and_run():
    # Initialize the ROS node
    rospy.init_node('key_publisher_node', anonymous=True)
    
    # Set up the publishers
    reset_pub = rospy.Publisher('/reset_command', Int32, queue_size=10)
    run_pub = rospy.Publisher('/env_run', Int32, queue_size=10)
    
    rospy.loginfo("Press 's' to publish to /reset_command and 'z' to publish to /env_run.")
    rospy.loginfo("Press 'h' to exit the node.")

    def on_press(key):
        try:
            if key.char == 's':
                # Publish a message to /reset_command when 's' is pressed
                reset_msg = Int32()
                reset_msg.data = 1  # You can modify this value as needed
                reset_pub.publish(reset_msg)
                rospy.loginfo("Published to /reset_command: %d" % reset_msg.data)

            elif key.char == 'z':
                # Publish a message to /env_run when 'z' is pressed
                run_msg = Int32()
                run_msg.data = 1  # You can modify this value as needed
                run_pub.publish(run_msg)
                rospy.loginfo("Published to /env_run: %d" % run_msg.data)

            elif key.char == 'h':
                # Exit the node when 'c' is pressed
                rospy.loginfo("Exiting the node...")
                return False  # This stops the listener

        except AttributeError:
            # Handle special keys if needed
            pass

    # Set up the listener
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  # Keeps the listener active

if __name__ == '__main__':
    try:
        publish_reset_and_run()
    except rospy.ROSInterruptException:
        pass

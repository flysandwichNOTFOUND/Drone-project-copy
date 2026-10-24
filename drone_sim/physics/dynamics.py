"""Shared flight state for the Gym/PyBullet backend."""

import numpy as np


class DroneState:
    def __init__(self):

        self.position = np.zeros(3, dtype = float)          # x, y, z

        self.velocity = np.zeros(3, dtype = float)          # vx, vy, vz

        self.orientation = np.zeros(3, dtype = float)       # roll, pitch, yaw

        self.angular_velocity = np.array([0, 0, 0], dtype = float)  # p, q, r

        self.motor_speeds = np.array([0, 0, 0, 0], dtype = float)      # w1, w2, w3, w4
        self.ground_contact = False
        self.contact_speed = 0.0

    def copy(self):

        copied_state = DroneState()

        copied_state.position = self.position.copy()
        copied_state.velocity = self.velocity.copy()
        copied_state.orientation = self.orientation.copy()
        copied_state.angular_velocity = self.angular_velocity.copy()
        copied_state.motor_speeds = self.motor_speeds.copy()
        copied_state.ground_contact = self.ground_contact
        copied_state.contact_speed = self.contact_speed

        return copied_state

    def reset(self):

        self.position.fill(0)
        self.velocity.fill(0)
        self.orientation.fill(0)
        self.angular_velocity.fill(0)
        self.motor_speeds.fill(0)
        self.ground_contact = False
        self.contact_speed = 0.0

    def print_status(self):

        print("Drone Status")
        print("------------")
        print("Position:", self.position)
        print("Velocity:", self.velocity)
        print("Orientation:", self.orientation)
        print("Angular Velocity:", self.angular_velocity)
        print("Motor Speeds:", self.motor_speeds)


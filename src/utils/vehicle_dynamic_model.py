#!/usr/bin/env python
import numpy as np
import pandas as pd
import queue
TRACTION = 'traction'
A_REPORT = 'a_report'
PITCH = 'pitch_pose'
YAW_RATE = 'yaw_rate'
STEERING_ANGLE = 'steering_angle'
V = 'v_current_dbw'
BENCHMARK = 'benchmark'

THROTTLE = 'throttle'
GEAR_RATIO = 'gear_ratio'
FRICTION = 'friction'
MASS = 'mass'
DRAG_COEFFICIENT = 'drag_coefficient'
ROLLING_COEFFICIENT = 'rolling_coefficient'
RAW_INPUT = [
    THROTTLE,
    GEAR_RATIO,
    FRICTION,
    MASS,
    PITCH,
    DRAG_COEFFICIENT,
    ROLLING_COEFFICIENT
]

OUTPUT_DATA = [
    A_REPORT,
    TRACTION,
    PITCH,
    V,
    YAW_RATE,
    STEERING_ANGLE,
    BENCHMARK
]

class VehicleDynamicModel():
    def __init__(self, config, init_v = 0.0):
        self._config = config
        self.reset(init_v)
    def reset(self, init_v = 0.0):
        self._v = init_v
        self._acc_history = queue.Queue()
        self._acc_signal = 0.0
    
    def get_state(self):
        data = {
            TRACTION: self.compute_traction(self._throttle,self._gear_ratio,self._friction_torque),
            A_REPORT: self._acc_signal,
            PITCH: self._pitch,
            YAW_RATE: 0.0,
            STEERING_ANGLE: 0.0,
            V: self._v,
            BENCHMARK: self._acc_signal
        }
        return pd.DataFrame(data, index = [0], columns=OUTPUT_DATA)
    
    def update(self,u):
        self._throttle = u[THROTTLE]
        self._gear_ratio = u[GEAR_RATIO]
        self._friction_torque =u[FRICTION]
        self._vehicle_mass = u[MASS]
        self._pitch = u[PITCH]
        self._drag_coefficient = u[DRAG_COEFFICIENT]
        self._rolling_friction_coefficient = u[ROLLING_COEFFICIENT]
        acc = self.compute_acc(self._throttle, self._gear_ratio, self._friction_torque, self._v, self._vehicle_mass, self._pitch, self._drag_coefficient, self._rolling_friction_coefficient)
        self._acc_history.put(acc)
        if(self._acc_history.qsize() >= 10):
            self._acc_signal = self._acc_history.get()
        self._v += acc *  self._config['sample_time']
        if(self._v < 0.0):
            self._v = 0.0   

    def compute_acc(self, throttle, gear_ratio, friction_torque, v, vehicle_mass, pitch, drag_coefficient = None, rolling_friction_coefficient = None):
        if(drag_coefficient == None):
            self._drag_coefficient = self._config['drag_coefficient']
        else:
            self._drag_coefficient = drag_coefficient
        if(rolling_friction_coefficient == None):
            self._rolling_friction_coefficient = self._config['rolling_friction_coefficient']
        else:
            self._rolling_friction_coefficient = rolling_friction_coefficient

        overall_ratio = self.compute_overall_ratio(gear_ratio)
        M_eq = self.compute_equivalent_mass(vehicle_mass, overall_ratio)
        F_aero = self.compute_aero_drag(v)
        F_roll = self.compute_rolling_resis(vehicle_mass)
        F_grav = self.compute_gravity_load(vehicle_mass, pitch)
        return (((throttle - friction_torque) * self._config['max_throttle_engine_torque'] * overall_ratio *
                self._config['transmission_efficiency'] / self._config['effective_tire_radius']) -
                F_aero - F_roll - F_grav) / M_eq
    
    def compute_traction(self,throttle, gear_ratio, friction_torque):
        overall_ratio = self.compute_overall_ratio(gear_ratio)
        return (throttle - friction_torque) * overall_ratio * self._config['max_throttle_engine_torque']  * self._config['transmission_efficiency'] / self._config['effective_tire_radius'] 
    
    def compute_overall_ratio(self, gear_ratio):
        overall_ratio = gear_ratio * self._config['axle_drive_ratio']
        if(type(overall_ratio) != float):
            overall_ratio = overall_ratio.fillna(0)
        return overall_ratio

    def compute_equivalent_mass(self, vehicle_mass, overall_ratio):
        rotation_equivalent_mass = (self._config['inertia_wheels'] + self._config['inertia_engine'] * self._config['transmission_efficiency'] * overall_ratio**2) / (self._config['effective_tire_radius']**2)
        return vehicle_mass + rotation_equivalent_mass
    
    def compute_aero_drag(self, v):
        return 0.5 * self._drag_coefficient * self._config['vehicle_frontal_area'] * self._config['air_density'] * v * v

    def compute_rolling_resis(self, vehicle_mass):
        return self._rolling_friction_coefficient * vehicle_mass * self._config['gravity_acceleration']
    
    def compute_gravity_load(self, vehicle_mass, pitch):
        return vehicle_mass * self._config['gravity_acceleration'] * np.sin(-pitch)

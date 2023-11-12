#!/usr/bin/env pybricks-micropython
import random
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

# from random import randint, choice

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.

ev3 = EV3Brick()

# Defining Ports
LEFT_MOTOR = Port.B
RIGHT_MOTOR = Port.C
LIGHT_SENSOR = Port.S1
OBSTACLE_SENSOR = Port.S4

# Defining Robot Parameters
WHITE_VALUE = 29
BLACK_VALUE = 8
TURN_ANGLE = 6
DRIVE_SPEED = 40
WHEEL_DIAMETER = 56 #55.5
AXLE_TRACK = 227 #104 

# Defining Q-learning Parameters
GAMMA = 0.8
BETA = 0.5 #discount rate
EPISODES = 10

NUM_STATES = 3 # (0 - Out,  1 - Margin, 2 - In)
NUM_ACTIONS = 3 # (0 - MoveForward,  1 - TurnLeft, 2 - TurnRight, 3 - MoveBackward)

FILE_PATH = 'qTable.txt'

# Defining Motors and Sensors
leftMotor = Motor(LEFT_MOTOR)
rightMotor = Motor(RIGHT_MOTOR)
lightSensor = ColorSensor(LIGHT_SENSOR)
obstacleSensor = InfraredSensor(OBSTACLE_SENSOR)

# Initializing the Robot Instance
robot = DriveBase(leftMotor, rightMotor, WHEEL_DIAMETER, AXLE_TRACK)

QTable = [[0] * NUM_ACTIONS for _ in range(NUM_STATES)]
# QTable = [[0,0,250],[0,250,0],[250,0,0]]

# Function to write a 2D array to a file
def write_2d_array_to_file(file_path, data):
    with open(file_path, 'w') as file:
        for row in data:
            line = ' '.join(map(str, row))  # Convert each element to a string and join with spaces
            file.write(line + '\n')  # Write each row as a line in the file

# Function to read a 2D array from a file
def read_2d_array_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [float(x) for x in line.strip().split()]  # Split the line into integers
            data.append(row)
    return data

def printQTable():
    print("Actions\t\t|", end="")  
    print("0 (L)", end="\t")
    print("1 (F)", end="\t")
    print("2 (R)", end="\t")
    print("\n------------------------")

    for state in range(NUM_STATES):
        print("State",state, "|", end="")
        for action in range(NUM_ACTIONS):
            print(round(QTable[state][action],2) , end="\t")
        print("\n")
    print("\n------------------------")
    print("------------------------\n")


    
def moveForward(speed):
  robot.straight(speed)

def moveBackward(speed):
    # while sr > BLACK_VALUE and sr < WHITE_VALUE:
    robot.straight(-speed)

def turnRight(angle):
  robot.turn(angle)

def turnLeft(angle):
  robot.turn(-angle)

def setState(sr):
    if sr < BLACK_VALUE:
        return 0
    elif sr >= BLACK_VALUE and sr <= WHITE_VALUE:
        return 1
    elif sr > WHITE_VALUE:
        # print(sr)
        return 2


def executeActionTestPhase(action):
    # sr = lightSensor.reflection()
    # print(action,',',sr)
    if action == 0:
        turnLeft(TURN_ANGLE)
    elif action == 1:
        moveForward(DRIVE_SPEED)
    elif action == 2:
        turnRight(TURN_ANGLE)
    elif action == 3:
        moveBackward(DRIVE_SPEED)
        
def test():
    # Testing the learned policy

    # following line can read a text document of the qtable
    global QTable
    QTable = read_2d_array_from_file(FILE_PATH)
    printQTable()
    while True:
        sr = lightSensor.reflection()
        
        state = setState(sr)
        print('State:',state)
        
        # if state == 2:
        #     break

        action = QTable[state].index(max(QTable[state]))
        print("act: ",action)
        # wait(1000)
        executeActionTestPhase(action)
        # newState, reward = executeAction(state, action)
        # state = newState


# sensorRead()
# qlearn()

# End of Learning Phase
ev3.speaker.beep(500, 500)

test()

# End of Testing Phase
ev3.speaker.beep(500, 500)

# Stop the robot
leftMotor.stop()
rightMotor.stop()


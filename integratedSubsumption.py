#!/usr/bin/env pybricks-micropython
import pickle
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

ev3 = EV3Brick()

# Defining Ports
LEFT_MOTOR = Port.B
RIGHT_MOTOR = Port.C
LIGHT_SENSOR = Port.S1
OBSTACLE_SENSOR = Port.S4

# Defining Robot Parameters
WHITE_VALUE = 28
BLACK_VALUE = 8
TURN_ANGLE = 5
DRIVE_SPEED = 10
WHEEL_DIAMETER = 56  # 55.5
AXLE_TRACK = 227  # 104
DISTANCE_TO_OBSTACLE = 25

# Defining Q-learning Parameters
ALPHA = 0.2  # Learning rate
GAMMA = 0.8  # Discount rate
EPISODES = 20

STATES = ["black", "margin", "white"]
NUM_STATES = len(STATES)
ACTIONS = ["left", "forward", "right"]
NUM_ACTIONS = len(ACTIONS)

# CONFIG_0 = [(black,right,margin), (white,left,margin),(margin,right,white),(margin,left,black)]
CONFIG_0 = [(0, 2, 1), (2, 0, 1), (1, 2, 2), (1, 0, 0)]
# CONFIG_1 = [(white, right, margin), (black, left, margin),(margin,right,black),(margin,left,white)]
CONFIG_1 = [(2, 2, 1), (0, 0, 1), (1, 2, 0), (1, 0, 2)]

# Defining Motors and Sensors
leftMotor = Motor(LEFT_MOTOR)
rightMotor = Motor(RIGHT_MOTOR)
lightSensor = ColorSensor(LIGHT_SENSOR)
obstacleSensor = InfraredSensor(OBSTACLE_SENSOR)

# Initializing the Robot Instance
robot = DriveBase(leftMotor, rightMotor, WHEEL_DIAMETER, AXLE_TRACK)

QTable = [[0] * NUM_ACTIONS for _ in range(NUM_STATES)]
filePath = 'qTable.pkl'



# Returns current state
def getState():
    sr = lightSensor.reflection()
    if sr < BLACK_VALUE:
        return 0
    elif sr >= BLACK_VALUE and sr <= WHITE_VALUE:
        return 1
    elif sr > WHITE_VALUE:
        return 2

def executeActionTest(action):
    if action == 0:
        robot.drive(0, -50)
    elif action == 1:
        robot.drive(150, 0)
    elif action == 2:
        robot.drive(0, 50)

def moveForward(speed):
    robot.straight(speed)

# def moveBackward(speed):
#     while getState() == 1:
#         robot.straight(-speed)

def turnRight(angle):
    robot.turn(angle)

def turnLeft(angle):
    robot.turn(-angle)

def getConfig():
    turnLeft(35)
    l = getState()
    turnRight(70)
    r = getState()
    turnLeft(35)

    if (l, r) == (0, 2):
        return 0
    elif (l, r) == (2, 0):
        return 1
    else:
        return None

# Updates and returns action based on config
def configProof(action, config):
    if action == 1:
        return action

    elif config == 0:
        return action

    elif config == 1:
        if action == 0:
            return 2
        elif action == 2:
            return 0

# Loads Q Table from a file
def loadQTable():
    print("Loading Q Table")
    with open(filePath, 'rb') as file:
        loadedData = pickle.load(file)
    return loadedData

def printQTable():
    print("Printing Q Table")
    for row in QTable:
        print(row)
    print("------------------------\n")

def line_following_behavior():
    global suppressed

    QTable = loadQTable()
    printQTable()

    config = getConfig()
    if config == None:
        return

    ev3.speaker.beep(500, 500)


    while True:
        if obstacleSensor.distance() < DISTANCE_TO_OBSTACLE:
          suppressed = True
        if not suppressed:  # Adjust this threshold as needed
            #line follower with q table
            state = getState()
            action = configProof(QTable[state].index(max(QTable[state])), config)
            executeActionTest(action)
        else:
            break

def obstacle_avoidance_behavior():
    global suppressed
    while True:
        if obstacleSensor.distance() < DISTANCE_TO_OBSTACLE:  # Obstacle detected, adjust this distance as needed
            robot.stop()
            robot.straight(-DRIVE_SPEED)
            robot.turn(180)
        else:
            suppressed = False
            break  # No obstacle detected

# Main control loop
while True:
    if obstacleSensor.distance() < DISTANCE_TO_OBSTACLE:  # Obstacle detected, higher priority
        obstacle_avoidance_behavior()
    else:
        line_following_behavior()

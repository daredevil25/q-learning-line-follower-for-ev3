#!/usr/bin/env pybricks-micropython
import random
import json
import time 
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
TURN_ANGLE = 20
DRIVE_SPEED = 20
WHEEL_DIAMETER = 56 #55.5
AXLE_TRACK = 227 #104 

# Defining Q-learning Parameters
ALPHA = 0.2 # Learning rate 
GAMMA = 0.8 # Discount rate
EPISODES = 20

STATES = ["black", "margin", "white"]
NUM_STATES = len(STATES)
ACTIONS = ["left", "forward", "right"]
NUM_ACTIONS = len(ACTIONS)

# MODE_0 = [(0,)]
# Defining Motors and Sensors
leftMotor = Motor(LEFT_MOTOR)
rightMotor = Motor(RIGHT_MOTOR)
lightSensor = ColorSensor(LIGHT_SENSOR)
obstacleSensor = InfraredSensor(OBSTACLE_SENSOR)

# Initializing the Robot Instance
robot = DriveBase(leftMotor, rightMotor, WHEEL_DIAMETER, AXLE_TRACK)
mode = 0

QTable = [[0] * NUM_ACTIONS for _ in range(NUM_STATES)]

filePath = './qTable.json'

def writeQtable():
    print("Writing")
    with open(filePath, 'w') as file:
        json.dump(QTable, file)

def loadQTable():
    with open(filePath, 'r') as file:
        QTable = json.load(file)

def printQTable():
    print(QTable)

    # print("Actions\t\t|", end="")  
    # print("0 (L)", end="\t")
    # print("1 (F)", end="\t")
    # print("2 (R)", end="\t")
    # print("\n------------------------")
    
    # for state in range(NUM_STATES):
    #     print("State",state, "|", end="")
    #     for action in range(NUM_ACTIONS):
    #         print(round(QTable[state][action],2) , end="\t")
    #     print("\n")
    # print("\n------------------------")
    # print("------------------------\n")

def modeProof(action):
    if action == 1: return action
    
    elif mode == 0:
        return action

    elif mode == 1:
        if action == 0: return 2
        elif action == 2: return 0

# Function to choose an action based on epsilon-greedy policy
def pickAction(state, epsilon):
    # Exploration: Choose a random action
    if random.random() < epsilon:
        action = random.randint(0, NUM_ACTIONS - 1)
        print("Random: ", ACTIONS[action])
        return action
    # Exploitation: Choose the action with the highest Q-value
    else:
        action = modeProof(QTable[state].index(max(QTable[state])))
        print("Table: ", ACTIONS[action])
        return action

def moveForward(speed):
    while getState() == 1:
        robot.straight(speed)

def moveBackward(speed):
    while getState() == 1:
        robot.straight(-speed)

def turnRight(angle):
    while getState() != 1:
        robot.turn(angle)

def turnLeft(angle):
    while getState() != 1:
        robot.turn(-angle)

# function to execute an action and return the next state and reward
def executeAction(action):
    if action == 0:
        turnLeft(TURN_ANGLE)
    elif action == 1:
        moveForward(DRIVE_SPEED)
    elif action == 2:
        turnRight(TURN_ANGLE)
    elif action == 3:
        moveBackward(DRIVE_SPEED)

    # Update state based on the light sensor reading
    newState = getState()

    # Define rewards
    if newState == 0:
        reward = -10
    elif newState == 1:
        reward = 50
    else:
        reward = -10

    return newState, reward

# Returns current state
def getState():
    sr = lightSensor.reflection()
    if sr < BLACK_VALUE:
        return 0
    elif sr >= BLACK_VALUE and sr <= WHITE_VALUE:
        return 1
    elif sr > WHITE_VALUE:
        return 2

# Update Q-value using Q-learning formula
def updateQTable(prevState, newState, action, reward):
    maxNextQ = max(QTable[newState])
    prev = QTable[prevState][action] 
    QTable[prevState][action] += ALPHA * (reward + GAMMA * maxNextQ - QTable[prevState][action])
    print("Q table: ", prev ,"->", QTable[prevState][action])


def qlearn(): 
    ev3.speaker.beep(500, 500)

    # Main Q-learning loop
    for episode in range(EPISODES):
        print("EPISODE", episode)
        # Decrease epsilon over time for exploration-exploitation trade-off
        epsilon = 1.0 / (episode + 1)
        print("EPSILON", epsilon)
        state = getState()
        totalReward = 0

        while totalReward <= 300:  
            print("---------------------")
            print("Current state: ", STATES[state])
            action = pickAction(state, epsilon)
            newState, reward = executeAction(action)
            print("New state: ", STATES[newState])
            print("Reward:", reward)

            updateQTable(state, newState, action, reward)

            state = newState
            totalReward += reward
            
            print("---------------------")
            # time.sleep(2)

        
        # print(f"Episode {episode + 1}: Total Reward = {totalReward}")
        # printQTable()
        # epsilon = epsilon * 2.1732 **(episode/EPISODES)
    writeQtable()

    ev3.speaker.beep(1000, 500)

def sensorRead():
    ev3.speaker.beep(500, 500)

    while True:
        # Read the light intensity
        lightIntensity = lightSensor.reflection()
        # Print the reading
        print("Light Intensity:", lightIntensity)
        # Print the value to the screen
        ev3.screen.draw_text(30, 40, lightIntensity)
        # Delay for a while (e.g., 1 second) to avoid excessive screen updates
        time.sleep(1)
        ev3.screen.clear()

# Testing the learned policy
def test():
    loadQTable()
    ev3.speaker.beep(500, 500)

    while True:
        state = getState()
        action = modeProof(QTable[state].index(max(QTable[state])))
        executeAction(action)

# sensorRead()
qlearn()
# test()

# Stop the robot
# leftMotor.stop()
# rightMotor.stop()

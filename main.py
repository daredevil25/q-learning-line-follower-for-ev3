#!/usr/bin/env pybricks-micropython
import random
import pickle
import time
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.

ev3 = EV3Brick()

# Defining Ports
LEFT_MOTOR = Port.D
RIGHT_MOTOR = Port.A
LIGHT_SENSOR = Port.S4
OBSTACLE_SENSOR = Port.S1

# Defining Robot Parameters
WHITE_VALUE = 18
BLACK_VALUE = 4
TURN_ANGLE = 5
DRIVE_SPEED = 10
WHEEL_DIAMETER = 56
AXLE_TRACK = 110
DISTANCE_TO_OBSTACLE = 25

# Defining Q-learning Parameters
ALPHA = 0.2  # Learning rate
GAMMA = 0.8  # Discount rate
EPISODES = 20 # Learning phases

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

# Initializing
suppressed = False
config = None
preForward = False
forwardSpeed = 5
filePath = 'qTable.pkl'

# Initializing the Robot Instance
robot = DriveBase(leftMotor, rightMotor, WHEEL_DIAMETER, AXLE_TRACK)

# Writes Q Table to a file
def writeQtable(writeData):
    print("Writing Q Table")
    with open(filePath, 'wb') as file:
        pickle.dump(writeData, file)

# Loads Q Table from a file
def loadQTable():
    print("Loading Q Table")
    with open(filePath, 'rb') as file:
        loadedData = pickle.load(file)
    return loadedData

# Prints the Q Table
def printQTable():
    print("Printing Q Table--------")
    for row in QTable:
        print(row)
    print("------------------------\n")

# Returns current state
def getState():
    sr = lightSensor.reflection()
    if sr < BLACK_VALUE:
        return 0
    elif sr >= BLACK_VALUE and sr <= WHITE_VALUE:
        return 1
    elif sr > WHITE_VALUE:
        return 2

def moveForward(speed):
    robot.straight(speed)

def moveBackward(speed):
    robot.straight(-speed)

def turnRight(angle):
    robot.turn(angle)

def turnLeft(angle):
    robot.turn(-angle)

# Returns the config of robot
def getConfig():
    turnLeft(40)
    l = getState()
    turnRight(80)
    r = getState()
    turnLeft(40)

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

# Check the availability of an object
def obstacleAvailable():
    obstacleStatus = obstacleSensor.distance() < DISTANCE_TO_OBSTACLE
    if (obstacleStatus): print("Obstacle detected")
    return obstacleStatus

# Picks an action based on epsilon-greedy policy
def pickAction(state, epsilon, config):
    # Exploration: Choose a random action
    if random.random() < epsilon:
        action = random.randint(0, NUM_ACTIONS - 1)
        print("random: ", end="")
        return action
    # Exploitation: Choose the action with the highest Q-value
    else:
        action = configProof(QTable[state].index(max(QTable[state])), config)
        print("table: ", end="")
        return action

# Execute action and return the next state and reward in learning phase
def executeActionLearn(action, state):
    count = 1
    if action == 0:
        while getState() == state:
            turnLeft(TURN_ANGLE)
            count+=1

    elif action == 1:
        moveForward(DRIVE_SPEED)
        count+=1

    elif action == 2:
        while getState() == state:
            turnRight(TURN_ANGLE)
            count+=1

    newState = getState()

    # Define rewards
    if newState == 0 or newState == 2:
        reward = -10
    elif newState == 1:
        reward = 60/count

    return newState, reward

# Execute actions in testing phase
def executeActionTest(action):
    if action == 0:
        robot.drive(0, -80)
    elif action == 1:
        robot.drive(300, 0)
    elif action == 2:
        robot.drive(0, 80)

# Update Q-value using Q-learning formula
def updateQTable(prevState, newState, action, reward, config):
    action = configProof(action, config)
    maxNextQ = max(QTable[newState])
    tableUpdate = ALPHA * (reward + GAMMA * maxNextQ -
                           QTable[prevState][action])
    QTable[prevState][action] += tableUpdate
    print("Table change: ", tableUpdate)
    print("updated-> ", prevState, action)

# Main Q-learning function
def qlearn():
    global QTable 
    QTable = [[0] * NUM_ACTIONS for _ in range(NUM_STATES)]
    ev3.speaker.beep(500, 500)

    epsilon = 1.0
    config = getConfig()
    if config == None:
        return

    # Main Q-learning loop
    for episode in range(EPISODES):
        print("EPISODE", episode)
        print("EPSILON", epsilon)

        state = getState()
        totalReward = 0

        while totalReward <= 300:
            print("---------------------")
            print("Config   ", config)
            action = pickAction(state, epsilon, config)
            newState, reward = executeActionLearn(action, state)
            print(STATES[state], "->", ACTIONS[action], "->",
                  STATES[newState], ": Reward", reward, end=" ")

            updateQTable(state, newState, action, reward, config)

            transition = (state, action, newState)
            if (transition in CONFIG_0):
                config = 0
            elif (transition in CONFIG_1):
                config = 1

            state = newState
            totalReward += reward
            print("Total Reward:", totalReward)

            print("---------------------")

        printQTable()
        writeQtable(QTable)
        epsilon -= 1 / (EPISODES) # Decrease epsilon over time for exploration-exploitation trade-off

    ev3.speaker.beep(1000, 500)

# Repeatedly read sensor reading for initial configuration
def sensorRead():
    ev3.speaker.beep(500, 500)
    while True:
        lightIntensity = lightSensor.reflection()
        print("Light Intensity:", lightIntensity)
        ev3.screen.draw_text(30, 40, lightIntensity) # Print the value to the screen
        time.sleep(1) # Delay for a while to avoid excessive screen updates
        ev3.screen.clear()

# Follows line until an obstacle is detected
def lineFollowingBehavior():
    print('Line following')
    global config
    global suppressed
    global QTable

    while not suppressed:   
        state = getState()
        action = configProof(QTable[state].index(max(QTable[state])), config)
        executeActionTest(action)

        if obstacleAvailable():
            suppressed = True   

# Avoids obstacle by taking a 180 degree turn
def obstacleAvoidanceBehavior():
    ev3.speaker.beep(700, 300) 
    print("Obstacle avoiding")
    global config
    global suppressed

    while True:
        robot.stop()
        moveBackward(100)
        turnRight(360)

        config = 0 if config == 1 else 1

        if not obstacleAvailable(): 
            suppressed = False
            break   

    ev3.speaker.beep(700, 300) 

# Main control loop
def test():
    global QTable
    global config

    QTable = loadQTable()
    printQTable()

    config = getConfig()
    if config == None:
        print("Config error")
        return

    ev3.speaker.beep(500, 500)

    while True:
        if obstacleAvailable():
            obstacleAvoidanceBehavior()
        else:
            lineFollowingBehavior()

# sensorRead()
# qlearn()
test()

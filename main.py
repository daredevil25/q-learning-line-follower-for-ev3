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
LIGHT_SENSOR = Port.S3
OBSTACLE_SENSOR = Port.S2

# Defining Robot Parameters
WHITE_VALUE = 29
BLACK_VALUE = 8
TURN_ANGLE = 5
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

# Function to choose an action based on epsilon-greedy policy
def pickAction(state, epsilon):
    # Exploration: Choose a random action
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    # Exploitation: Choose the action with the highest Q-value
    else:
        return QTable[state].index(max(QTable[state]))

def moveForward(speed):
    sr = lightSensor.reflection()
    print("GGGGGGGG", sr)
    # while True:
    while sr > BLACK_VALUE and sr < WHITE_VALUE:
        print(sr)
        robot.straight(speed)
        sr = lightSensor.reflection()
    return

def moveBackward(speed):
    # while sr > BLACK_VALUE and sr < WHITE_VALUE:
    robot.straight(-speed)

def turnRight(angle):
    sr = lightSensor.reflection()
    while sr < BLACK_VALUE or sr > WHITE_VALUE:
        robot.turn(angle)
        sr = lightSensor.reflection()
        
    return

def turnLeft(angle):
    sr = lightSensor.reflection()
    while sr < BLACK_VALUE or sr > WHITE_VALUE:
        robot.turn(-angle)
        sr = lightSensor.reflection()
    return

# function to execute an action and return the next state and reward
def executeAction(state, action):
    sr = lightSensor.reflection()
    # print(sr)
    if action == 0:
        turnLeft(TURN_ANGLE)
    elif action == 1:
        moveForward(DRIVE_SPEED)
    elif action == 2:
        turnRight(TURN_ANGLE)
    elif action == 3:
        moveBackward(DRIVE_SPEED)

    # Update state based on the light sensor reading
    newState = setState(sr)

    # Define rewards
    if newState == 0:
        reward = -10
    elif newState == 1:
        reward = 50
    else:
        reward = -10

    return newState, reward

# Check State
def setState(sr):
    if sr < BLACK_VALUE:
        return 0
    elif sr >= BLACK_VALUE and sr <= WHITE_VALUE:
        return 1
    elif sr > WHITE_VALUE:
        # print(sr)
        return 2

# Start of Learning Phase
ev3.speaker.beep(1000, 500)

def qlearn(): 
    # Main Q-learning loop
    for episode in range(EPISODES):
        print("EPISODE", episode)
        # Decrease epsilon over time for exploration-exploitation trade-off
        # epsilon = 1.0 / (episode + 1)
        epsilon = 0.99
        epsilon -= - 0.01 * episode
        state = setState(lightSensor.reflection())
        totalReward = 0

        while totalReward <= 300 and totalReward >= -300:  
            action = pickAction(state, epsilon)
            newState, reward = executeAction(state, action)

            # Update Q-value using Q-learning formula
            maxNextQ = max(QTable[newState])
            QTable[state][action] = (1-GAMMA) * QTable[state][action] + GAMMA * (reward + maxNextQ * BETA)

            state = newState
            totalReward += reward
            print(totalReward)

        # print(f"Episode {episode + 1}: Total Reward = {totalReward}")
        #following line writes the qtable to a text file
        printQTable()
    write_2d_array_to_file(FILE_PATH, QTable)
    

def sensorRead():
    while True:
        # Read the light intensity
        lightIntensity = lightSensor.reflection()
        # Print the reading
        print("Light Intensity:", lightIntensity)
        # Print the value to the screen
        ev3.screen.draw_text(30,40,lightIntensity)
        wait(10)
        ev3.screen.clear()


def executeActionTestPhase(action):
    sr = lightSensor.reflection()
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
        wait(1000)
        executeActionTestPhase(action)
        # newState, reward = executeAction(state, action)
        # state = newState


# sensorRead()
qlearn()

# End of Learning Phase
ev3.speaker.beep(500, 500)

# test()

# End of Testing Phase
ev3.speaker.beep(500, 500)

# Stop the robot
leftMotor.stop()
rightMotor.stop()

import random
import time
import numpy as np
import gymnasium as gym
import ale_py as ale
from gymnasium.wrappers import GrayscaleObservation

import keras
from keras import layers

import numpy as np

class ReplayMemory:
    def __init__(self, maxSize=5000):
        self.buffer = []
        self.maxSize = maxSize

    def add(self, memory):
        if len(self.buffer) >= self.maxSize:
            self.buffer.pop()

        self.buffer.insert(0, memory)

    def sample(self, size):
        return random.sample(self.buffer, size)

    def size(self):
        return len(self.buffer)

class Agent:
    def __init__(self, env, epsilon=1.0, epsilonDecay=0.0001, epsilonMin=0.1, isTraining=True, gamma=0.9, learningRate=0.003, batchSize = 32, trainFreq = 5):

        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.isTraining = isTraining

        self.env = env

        self.gamma = gamma
        self.learningRate = learningRate

        # use this if we want deep q learning
        self.model = self.createAgent(learningRate)

        self.batchSize = batchSize
        self.trainFreq = trainFreq

    def createAgent(self, learningRate):
        model = keras.Sequential([
            layers.Dense(64, input_dim=1, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(5, activation="linear")
        ])

        # model = keras.Sequential([
        #     layers.Dense(32, input_dim=1, activation="relu"),
        #     layers.Dense(32, activation="relu"),
        #     layers.Dense(5, activation="linear")
        # ])

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learningRate))

        return model

    def changeEnv(self, env):
        self.env = env

    def preprocessState(self, state):
        state = state[30:-15]

        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] > 10:
                    state[i][j] = 1
                else:
                    state[i][j] = 0

        state = tuple(tuple(row) for row in state)

        hashVal = hash(state)

        # turn tuple into hash (I hope this works)
        hashVal = abs(hashVal % (165 * 160 * 2))

        return hashVal

    def chooseAction(self, state):
        # if we are training...
        if self.isTraining:

            # decay the epsilon value
            if self.epsilon > self.epsilonMin:
                self.epsilon -= self.epsilonDecay

            # roll a random number
            randnum = random.uniform(0, 1)

            # choose randomly if smaller than epsilon
            if randnum < self.epsilon:
                randaction = np.random.randint(0, 6)
                return randaction

        # get the maximum value
        qVals = self.model.predict(np.array(state))
        return np.argmax(qVals)


    def evaluate(self, numSteps=None, numEpisodes=1):

        Rewards = []

        self.isTraining = False
        for i in range(numEpisodes):
            state, _ = self.env.reset()
            state = self.preprocessState(state)
            stepCount = 0
            reward = 0

            done = False

            while not done:

                action = self.chooseAction(state)
                nextState, reward, done, _, _ = self.env.step(action)

                nextState = self.preprocessState(nextState)

                state = nextState

                if numSteps:
                    if numSteps <= stepCount:
                        print("Agent Failed to reach goal in time with a final score of {}".format(reward))
                        break
                    else:
                        stepCount += 1

            Rewards.append(reward)

        return sum(Rewards)/len(Rewards)


    def runQLearning(self, memory, numSteps=None, numEpisodes=1):

        Rewards = []

        lossValues = []

        self.isTraining = True
        for i in range(numEpisodes):
            state, _ = self.env.reset()
            state = self.preprocessState(state)
            stepCount = 0
            reward = 0

            totalReward = 0

            done = False

            count = 0

            while not done:
                action = self.chooseAction(state)
                nextState, reward, done, _, _ = self.env.step(action)

                memory.add((state, action, reward, nextState, done))

                self.env.printGrid(0)

                state = nextState

                totalReward += reward

                if memory.size() > self.batchSize and self.trainFreq < count:
                    count = 0
                    batch = memory.sample(self.batchSize)

                    states, actions, rewards, nextStates, dones = zip(*batch)

                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    nextStates = np.array(nextStates)
                    dones = np.array(dones)

                    target = rewards + self.gamma * np.max(self.model.predict(nextStates), axis=1) * (1 - dones)
                    qVals = self.model.predict(states)

                    for k in range(self.batchSize):
                        qVals[k][actions[k]] = target[k]

                    history = self.model.fit(states, qVals, epochs=1, verbose=0)

                    lossValues.append(history.history['loss'][0])

                if numSteps:
                    if numSteps <= stepCount:
                        print("Episode ended in {} steps with a final score of {}".format(numSteps, reward))
                        # time.sleep(1)
                        break
                    else:
                        stepCount += 1

            print("Epsilon Value: {}".format(self.epsilon))
            print("Total Reward This Episode: {}".format(totalReward))

            Rewards.append(reward)

            # time.sleep(1)

        return (sum(lossValues) / len(lossValues))

memory = ReplayMemory()

env = gym.make("SpaceInvadersDeterministic-v4", render_mode="human")
env = GrayscaleObservation(env)

agent = Agent(env, epsilon=1.0, epsilonDecay=0.00001, epsilonMin=0.1)

agent.runQLearning(memory, numEpisodes=100)

env = gym.make("SpaceInvadersDeterministic-v4", render_mode="human")
env = GrayscaleObservation(env)

agent.changeEnv(env)

agent.evaluate(numEpisodes=10)





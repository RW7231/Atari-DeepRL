import random
import time
import numpy as np
import gymnasium as gym
import ale_py as ale
from gymnasium.wrappers import GrayscaleObservation

class Agent:
    def __init__(self, env, epsilon=1.0, epsilonDecay=0.0001, epsilonMin=0.1, isTraining=True, gamma=0.9, learningRate=0.003):

        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.isTraining = isTraining

        self.env = env

        self.gamma = gamma
        self.learningRate = learningRate

        # shape of q value array is size of the states and actions possible
        self.qValues = np.zeros(shape=(165 * 160 * 2, 4))

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
                randaction = np.random.randint(0, 4)
                return randaction

        # get the maximum value
        qVals = np.array(self.qValues[state])
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


    def runQLearning(self, numSteps=None, numEpisodes=1):

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

            while not done:
                action = self.chooseAction(state)
                nextState, reward, done, _, _ = self.env.step(action)

                nextState = self.preprocessState(nextState)
                nextBestAction = np.argmax(np.array(self.qValues[nextState]))

                target = reward + self.gamma * self.qValues[nextState][nextBestAction]

                error = target - self.qValues[state][action]

                lossValues.append((error * error))

                self.qValues[state][action] += self.learningRate * error

                state = nextState

                totalReward += reward

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

env = gym.make("BreakoutNoFrameskip-v4")
env = GrayscaleObservation(env)

agent = Agent(env, epsilon=1.0, epsilonDecay=0.00001, epsilonMin=0.1)

agent.runQLearning(numEpisodes=100)

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = GrayscaleObservation(env)

agent.changeEnv(env)

agent.evaluate(numEpisodes=10)





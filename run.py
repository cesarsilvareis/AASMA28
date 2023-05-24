import argparse
import time

import numpy as np
from gym import Env

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper
from ambulance_emergency_response import AmbulanceERS
from snake_game import SnakeEnv


def run_single_agent(environment: Env, agent: Agent, n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminal = False
        # Reseting the environment (useful for comparated single consecutives autonomous agents)
        observation = environment.reset()
        while not terminal:
            steps += 1
            
            # time.sleep(1)
            agent.see(observation)
            agent_action = agent.action()
            next_observation, reward, terminal, info = environment.step(agent_action)
            
            print(f"Agent {agent.name} - Timestep {steps}")
            print(f"\tObservation: {observation}")
            # print(f"\tAction: {environment.get_action_meanings()[agent_action]}")
            print(f"\tInfo: {info}")
            print(f"\tReward: {reward}\n")

            environment.render()
            # time.sleep(1)

            observation = next_observation         


        results[episode] = steps

    environment.close()

    return results


class RandomAgent(Agent):

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__("Random Agent")
        self.n_actions = n_actions

    def action(self) -> int:
       return np.random.randint(self.n_actions)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = AmbulanceERS(

    )
    # environment = SingleAgentWrapper(environment, agent_id=0)

    environment.reset()

    while(True): pass

    # 2 - Setup agent
    agent = RandomAgent(environment.action_space.n)

    # 3 - Evaluate agent
    results = {
        agent.name: run_single_agent(environment, agent, opt.episodes)
    }

    # 4 - Compare results
    print(results)
    # compare_results(results, title="Random Agent on 'Predator Prey' Environment", colors=["orange"])


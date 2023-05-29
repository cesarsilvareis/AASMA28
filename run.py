import argparse
import time
import logging

import numpy as np
from gym import Env

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper
from ambulance_emergency_response.environment import AmbulanceERS
from ambulance_emergency_response.agents import GreedyAgent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s: %(message)s',)

def run_multiple_agents(environment: Env, agents: list(Agent), n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminal = False
        # Reseting the environment (useful for comparated single consecutives autonomous agents)
        observations = environment.reset()
        while not terminal:
            steps += 1
            
            time.sleep(1)

            agent_actions = []

            for i in range(len(agents)):
                agent = agents[i]
                agent.see(observations[i])
                agent_actions.append(agent.action())

            next_observations, nreward, terminal, ninfo = environment.step(agent_actions)
            
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = AmbulanceERS(
        city_size=(500, 500),
        num_agents=4,
        agent_coords=[(220, 40), (40, 220), (220, 440), (440, 220)],
        agent_num_ambulances=[2, 2, 2, 2]        
    )
    # environment = SingleAgentWrapper(environment, agent_id=0)

    environment.reset()

    environment.step([(1,(10, 10)), (0,), (0,), (0,)])  # TODO: actions: ASSIST(request)

    agent = GreedyAgent(environment.action_space.n)

    while(True):  # stop execution by closing window or Ctr^ C
        environment.render()
        environment.step([(0,), (0,), (0,), (0,)])

        time.sleep(0.5)

    # falta implementar o resto em baixo
    exit()

    # 2 - Setup agent
    agent = GreedyAgent(environment.action_space.n)

    # 3 - Evaluate agent
    results = {
        agent.name: run_single_agent(environment, agent, opt.episodes)
    }

    # 4 - Compare results
    print(results)
    # compare_results(results, title="Random Agent on 'Predator Prey' Environment", colors=["orange"])


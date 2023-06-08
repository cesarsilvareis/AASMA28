import argparse
import time
import logging

import numpy as np
from gym import Env

from aasma import Agent
from aasma.utils import compare_results, print_results
from ambulance_emergency_response.environment import AmbulanceERS
from ambulance_emergency_response.agents import RandomAgent, WeakGreedyAgent, StrongGreedyAgent, ConventionAgent, RoleAgent
from ambulance_emergency_response.settings import DEBUG, SEED, EXPERIMENT_FOLDER, OCCUPANCY_MAP_1, OCCUPANCY_MAP_2, OCCUPANCY_MAP_3, OCCUPANCY_MAP_4

if DEBUG:
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s: %(message)s',)

def run_multiple_agents(environment: Env, agents: list[Agent], n_episodes: int, render: bool=True) -> np.ndarray:

    results = {
        "Response-rate": np.zeros(n_episodes),
        "Response-time": np.zeros(n_episodes),
        "Ambulance-availability": np.zeros(n_episodes),
        "Resource-utilization": { agent.name : np.zeros(n_episodes) for agent in agents },
    }

    for episode in range(n_episodes):

        steps = 0
        terminal = False
        observations = environment.reset()
        while not terminal:
            steps += 1

            agent_actions = []

            for i in range(len(agents)):
                agent = agents[i]
                agent.see(observations[agent.name])
                agent_action = agent.action()
                agent_actions.append(agent_action)

            next_observations, nreward, terminal, ninfo = environment.step(agent_actions)

            if render:
                environment.render()
                time.sleep(.5)

            observations = next_observations

        # TODO: store a *relevant* metric of the episode
        results["Response-rate"][episode] = environment.metrics["Response-rate"]
        results["Response-time"][episode] = environment.metrics["Response-time"]
        results["Ambulance-availability"][episode] = environment.metrics["Ambulance-availability"]

        for agent in agents:
            results["Resource-utilization"][agent.name][episode] = environment.metrics["Resource-utilization"][agent.name]

    environment.close()

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--rendering", action=argparse.BooleanOptionalAction)
    parser.add_argument("--show-results", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = AmbulanceERS(
        city_size=(500, 500),
        num_agents=4,
        agent_coords=[(220, 40), (40, 220), (220, 440), (440, 220)],
        agent_num_ambulances=[1, 2, 2, 2],
        request_max_generation_steps=50,
        occupancy_map=OCCUPANCY_MAP_4,
        show_density_map=opt.show_results,
        save_density_filename=EXPERIMENT_FOLDER + "density_map.png" if opt.save_results else None,
        stressfulness=2
    )

    # sets seed
    # environment.seed(SEED)

    convention = [agency.name for agency in environment.agencies]

    # 2 - Setup agent teams
    teams = {
        "Random\n Agencies": [RandomAgent(agency.name, environment.N_AGENTS) for agency in environment.agencies],
        "Weak Greedy\n Agencies": [WeakGreedyAgent(agency.name, environment.N_AGENTS) for agency in environment.agencies],
        "Strong Greedy\n Agencies": [StrongGreedyAgent(agency.name, environment.N_AGENTS) for agency in environment.agencies],
        "Convention\n Agencies": [ConventionAgent(agency.name, environment.N_AGENTS, convention) for agency in environment.agencies],
        "Role\n Agencies": [RoleAgent(agency.name, environment.N_AGENTS) for agency in environment.agencies],
    }

    # 3 - Evaluate agent
    results = {
        "Response-rate": {},
        "Response-time": {},
        "Ambulance-availability": {},
        "Resource-utilization": {},
    }
    for team, agents in teams.items():
        print("Running %s team..." %team)
        result = run_multiple_agents(environment, agents, opt.episodes, opt.rendering)
        for metric in results.keys():
            results[metric][team] = result[metric]

    # 4 - Compare results
    print_results(results=results)

    if opt.show_results:
        for metric in results.keys():
            compare_results(results=results, title="Comparing the performance of all types of teams", 
                            metric=metric, colors=["orange", "green", "blue", "red", "purple"], 
                            filename=(EXPERIMENT_FOLDER + "%s.png" %metric) if opt.save_results else None)
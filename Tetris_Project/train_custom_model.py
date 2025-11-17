import numpy as np
import random
import os
import copy
import pickle
from custom_model import CUSTOM_AI_Model
from game import Game

CHECKPOINT_PATH = "checkpoint_custom_model.pkl"
BEST_AGENT_PATH = "10genmodel.pkl"


def train_custom_model(pop_size=50, generations=10, mutation_rate=0.2, save_every=10):
    print(f"Training custom algorithm with population size {pop_size}")

    start_gen = 0

    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint...")
        with open(CHECKPOINT_PATH, 'rb') as f:
            checkpoint = pickle.load(f)
        population = checkpoint['population']
        overall_best_agent = checkpoint['best_agent']
        overall_best_score = checkpoint['best_score']
        start_gen = checkpoint['generation'] + 1
    else:
        population = [CUSTOM_AI_Model() for _ in range(pop_size)]
        overall_best_agent = None
        overall_best_score = -1

    for gen in range(start_gen, generations):

        for agent in population:
            game = Game(mode="student", agent=agent)
            cleared = game.run_no_visual()
            agent.fit_score = cleared[1] 

        population.sort(key=lambda a: a.fit_score, reverse=True)
        top_agent = population[0]

        if overall_best_agent is None or top_agent.fit_score > overall_best_score:
            overall_best_score = top_agent.fit_score
            overall_best_agent = copy.deepcopy(top_agent)    

        elites = population[:3]

        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent = random.choice(elites)
            child = CUSTOM_AI_Model(
                genotype=parent.genotype.copy(),
                mutate=True,
                noise_sd=mutation_rate
            )
            new_population.append(child)

        population = new_population

        print(f"Gen {gen+1}/{generations}: Best this gen = {top_agent.fit_score}, Overall best = {overall_best_score}")

        if (gen + 1) % save_every == 0:
            checkpoint = {
                'generation': gen,
                'population': population,
                'best_agent': overall_best_agent,
                'best_score': overall_best_score
            }
            with open(CHECKPOINT_PATH, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"Checkpoint saved at generation {gen+1}")
            
    with open(BEST_AGENT_PATH, 'wb') as f:
        pickle.dump(overall_best_agent.genotype, f)
    print(f"Training finished! Best agent saved as {BEST_AGENT_PATH}")


if __name__ == "__main__":
    train_custom_model()

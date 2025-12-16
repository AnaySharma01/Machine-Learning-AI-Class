import torch
from custom_model import CUSTOM_AI_Model
from game import Game
import pickle
MODEL_PATH = "10genmodel.pkl"

def test_custom_model():
    device = "cpu" 
    print("Testing on device:", device)

    print("Loading saved agent...")
    with open(MODEL_PATH, "rb") as f:
        saved_agent = pickle.load(f)
    if isinstance(saved_agent, dict) and "genotype" in saved_agent:
        genotype = saved_agent["genotype"]
    else:
        genotype = saved_agent 

    agent = CUSTOM_AI_Model(genotype=genotype)

    game = Game(mode="student", agent=agent)
    score, cleared = game.run()

    print("Final score:", score)
    print("Lines cleared:", cleared)


if __name__ == "__main__":
    test_custom_model()

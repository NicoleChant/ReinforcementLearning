from agent import TheMatrixFactory
from maze import *

if __name__ == "__main__":
    factory = TheMatrixFactory()
    agent = factory.construct(agent = "QAgent" , current_state = (start_x , start_y))
    agent.qtrain()

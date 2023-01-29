from maze import *
from agent import *
import pygame
import threading
import logging


SURFACE_WIDTH = 400
screen = pygame.display.set_mode((WIDTH + SURFACE_WIDTH, HEIGHT))
pygame.display.set_caption("QMaze")

logging.basicConfig(filename = "training.log" ,
                    level = logging.INFO ,
                    format = "%(levelname)s:%(message)s")

def main():

    matrix = TheMatrixFactory()
    agent = matrix.construct(agent = "QAgent" , load_weights = False)

    agentBehavior = threading.Thread(target = agent.qtrain , daemon = True , args = (100,))
    agentBehavior.start()
    clock = pygame.time.Clock()

    run = True

    while run:
        ##reduce FPS
        clock.tick(FPS)
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        ##drawing our maze
        for x in range(len(maze)):
            for y in range(len(maze[0])):
                color = colors.get(maze[x][y], BLACK)
                pygame.draw.rect(screen, color, (x * CELL_WIDTH, y * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

        pygame.draw.line(screen , (255,255,255) , (WIDTH,0) , (WIDTH,HEIGHT))
        pygame.display.update()


if __name__ == "__main__":
    main()

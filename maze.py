import random

maze = []
BLOCK_CHANCE = 0.1
WIDTH = 800
HEIGHT = 800

## COLORS
BLACK = (0,0,0)
ORANGE = (100,100,50)
RED = (220,10,10)
BLUE = (20,50,220)

CELL_ROWS = 60
CELL_COLS = 60

CELL_WIDTH = WIDTH//CELL_ROWS
CELL_HEIGHT = HEIGHT//CELL_COLS

FPS = 30

for x in range(CELL_ROWS):
    maze.append([])
    for y in range(CELL_COLS):
        if random.random() <= BLOCK_CHANCE:
            maze[x].append("O")
        else:
            maze[x].append(0)

def generate_start():
    while True:
        x = random.randint(20,CELL_ROWS-1)
        y = random.randint(30,CELL_COLS-1)

        if maze[y][x] == 0:
            return x , y

food_x = 10
food_y = 8
maze[food_y][food_x] = "food"

#start_x , start_y = generate_start()
#maze[start_y][start_x] = "agent"

colors = {"O" : ORANGE ,
          "food" : RED ,
          "agent" : BLUE }

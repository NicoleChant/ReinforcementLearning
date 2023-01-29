from maze import *
from datetime import datetime
import logging
import time
import abc
from dataclasses import dataclass , field
from typing import Final , ClassVar
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict

@dataclass
class Agent:

    current_state : tuple[int,int] = field(init = False , repr = False , default = None)
    gathered_food : int = field(init = False , default = 0)
    actions : Final[list[str]] = field(init = False , default_factory = lambda : [">","v","^","<"])
    cooldown : Final[float] = 0.5
    history : list[tuple[int,int]] = field(init = False , repr = False)
    has_memory : bool = False

    def __post_init__(self) -> None:
        if self.has_memory:
            self.history = []

    def reset(self) -> None:
        if not self.current_state is None:
            maze[self.y][self.x] = 0
        start_x , start_y = generate_start()
        self.current_state = (start_x , start_y)
        maze[start_y][start_x] = "agent"
        maze[food_y][food_x] = "food"

    @property
    def x(self) -> int:
        return self.current_state[0]

    @property
    def y(self) -> int:
        return self.current_state[1]

    def succ_state(self , action : str) -> tuple[int,int]:
        ##παρούσα κατάσταση/θέση του πράκτορα στον λαβύρινθο
        x , y = self.current_state
        match action:
            case ">":
                x += 1
            case "<":
                x -= 1
            case "v":
                y += 1
            case "^":
                y -= 1
            case _:
                raise ValueError(f"Invalid action {action}.")
        return x , y

    def valid_actions(self) -> list[str]:
        return [action for action in self.actions if self.is_valid(*self.succ_state(action))]

    def update(self , x , y) -> bool:
        if self.is_valid(x,y):
            maze[self.y][self.x] = 0
            self.current_state = (x,y)
            return True
        return False

    def is_valid(self , x  : int , y : int) -> bool:
        return len(maze) > x >= 0 and len(maze[0]) > y >= 0 and maze[y][x] != "O"

    @abc.abstractmethod
    def get_action(self) -> str:
        """Implements AI Behavior"""
        pass

    def act(self) -> None:
        while True:

            action = self.get_action()
            x , y = self.succ_state(action)
            updated = self.update(x,y)
            if updated:
                if self.has_memory:
                    self.history.append((x,y))

                time.sleep(self.cooldown)

@dataclass
class RandomAgent(Agent):

    def get_action(self) -> str:
        return random.choice(self.valid_actions())


@dataclass
class QAgent(Agent):

    qtable : dict = field(init = False , repr = False , default_factory = lambda : defaultdict(dict))
    epsilon : Final[float] = 0.2
    discount : Final[float] = 0.9
    alpha : Final[float] = 0.9
    rewards : list[float] = field(init = False , repr = False , default_factory = list)
    load_weights : bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.load_weights:
            self.load_memory()
        else:
            self.init_qtable()

    def load_memory(self) -> None:
        with open("qtable.json" , mode = "r+" , encoding = "utf-8") as f:
            self.qtable = {eval(key) : value for key , value in json.load(f).items()}

    def is_terminal(self , x : int , y : int) -> bool:
        return maze[y][x] == "food"

    def init_qtable(self) -> None:
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                for action in self.actions:
                    self.qtable[(i,j)][action] = 0.0

    def get_action(self) -> str:
        """Returns QAgents action based on the ε-greedy strategy."""

        ##εξερεύνηση
        if random.random() <= self.epsilon:
            random_action = random.choice(self.valid_actions())
            return random_action
        ##εναντίον εκμετάλλευση //else δεν χρειάζεται αλλά για λόγους ομορφιάς αφήνεται εδώ λολ
        else:
            q_values = {action :self.qtable[self.succ_state(action)][action] for action in self.valid_actions()}
            return max(q_values , key = q_values.get)


    def qtrain(self  , episodes : int = 200 ,max_iter : int = 10000):
        for _ in tqdm(range(episodes)):
            self.reset()
            total_actions = 0

            while True:
                action = self.get_action()
                state = self.succ_state(action)
                reward = self.evaluate_Q(action , state)
                #self.rewards.append((total_actions , reward))
                total_actions += 1
                self.update(*state)

                if self.is_terminal(*self.current_state):
                    break
                else:
                    maze[self.y][self.x] = "agent"

                time.sleep(0.01)

                if total_actions > max_iter:
                    break

        self.save()

    def save(self) -> None:
        with open("qtable.json" , mode = "w+" , encoding = "utf-8") as f:
            json.dump({str(key) : value for key , value in self.qtable.items()} , f)

    def reward(self , state: tuple[int,int]) -> float:
        ##χρειάζεται a prior γνώση της τροφής αυτή η ευρεστική
        x , y = state

        ##ευκλείδεια απόσταση από την πηγή τροφής
        distance = ((food_y - y)**2 + (food_x - x)**2)**0.5
        return  - distance if distance > 0 else 1000.0

    def evaluate_Q(self , action : str , state : tuple[int,int]):
        reward = self.reward(state)
        maxQSPrime = max([self.qtable[self.succ_state(action)][future_action] for future_action in self.actions if \
                                self.is_valid(*self.succ_state(future_action))
                        ])

        ##εξίσωση bellman
        self.qtable[state][action] += self.alpha * (reward + self.discount * maxQSPrime - self.qtable[state][action])
        return reward

class TheMatrixFactory:

    def construct(self , agent : str , *args , **kwargs) -> Agent:
        return {"QAgent" : QAgent,
                "RandomAgent" : RandomAgent}[agent](*args , **kwargs)

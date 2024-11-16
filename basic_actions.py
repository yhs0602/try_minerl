import enum
import random

from numpy import argmax


class Index(enum.IntEnum):
    THIRST = 0
    HUNGER = 1
    ENERGY = 2
    HEALTH = 3


class Action(enum.IntEnum):
    DO = 0
    SLEEP = 1
    LOOK_UP = 2
    LOOK_DOWN = 3
    LOOK_LEFT = 4
    LOOK_RIGHT = 5


class OBS:
    def __init__(self, grid):
        self.grid = grid
        self.thirst = 0
        self.hunger = 0
        self.energy = 0
        self.health = 0


def norm(x, y):
    return (x**2 + y**2) ** 0.5


def inf_norm(x, y):
    return max(abs(x), abs(y))


class HeuristicAgent:
    def __init__(self, base_value):
        self.state = [0, 0, 0, 0]
        self.paths = []
        self.base_value = base_value
        self.position = (0, 0)

    def get_action(self, obs: OBS):
        self.state = [obs.thirst, obs.hunger, obs.energy, obs.health]
        differences = [(self.base_value[k] - self.state[k]) ** 2 for k in range(3)]
        # TODO: Apply weights
        most_prioritized = argmax(differences)
        match most_prioritized:
            case Index.THIRST:
                yield from self.drink_water(obs)
            case Index.HUNGER:
                yield from self.eat_cow(obs)
            case Index.ENERGY:
                yield from Action.SLEEP

    def drink_water(self, obs):
        water_location_delta = self.find_water(obs)
        if not water_location_delta:
            yield self.wander_around(obs)
        elif inf_norm(*water_location_delta) == 1:
            yield from self.look_at_water(obs)
            yield Action.DO
        else:
            self.paths.append(
                self.find_path(water_location_delta)
            )  # TODO: Discard old path
        yield from self.follow_path()

    def eat_cow(self, obs):
        cow_location_delta = self.find_cow(obs)
        if not cow_location_delta:
            pass
            # TODO("Wander around for a cow")
        elif inf_norm(*cow_location_delta) == 1:
            self.look_at_cow(obs)
            yield Action.DO
        else:
            self.paths.append(
                self.find_path(cow_location_delta)
            )  # TODO: Discard old path
        yield from self.follow_path()

    def follow_path(self):
        while self.paths:
            path = self.paths.pop()
            for direction in path:
                yield direction

    def find_path(self, delta):
        # TODO: Use RRT* or A* or dijkstra to find path
        pass

    # TODO: Consider obstacles for cost.
    # Generalize RRT* to find location with a certain property
    def find_water(self, obs):
        # assume obs has a grid info
        grid = obs.grid
        min_distance = float("inf")
        nearest_water = None
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y].tile == "water":
                    distance = abs(self.position[0] - x) + abs(self.position[1] - y)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_water = (x, y)
        return nearest_water

    def find_cow(self, obs):
        # assume obs has a grid info
        grid = obs.grid
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y].entity == "cow":
                    return x, y
        return None

    def look_at_water(self, obs):
        # find water in the grid and look at it
        grid = obs.grid
        if grid[self.position[0], self.position[1] + 1].tile == "water":
            yield Action.LOOK_DOWN
        elif grid[self.position[0], self.position[1] - 1].tile == "water":
            yield Action.LOOK_UP
        elif grid[self.position[0] + 1, self.position[1]].tile == "water":
            yield Action.LOOK_RIGHT
        elif grid[self.position[0] - 1, self.position[1]].tile == "water":
            yield Action.LOOK_LEFT
        yield Action.DO

    def look_at_cow(self, obs):
        # find cow in the grid and look at it
        grid = obs.grid
        if grid[self.position[0], self.position[1] + 1].entity == "cow":
            yield Action.LOOK_DOWN
        elif grid[self.position[0], self.position[1] - 1].entity == "cow":
            yield Action.LOOK_UP
        elif grid[self.position[0] + 1, self.position[1]].entity == "cow":
            yield Action.LOOK_RIGHT
        elif grid[self.position[0] - 1, self.position[1]].entity == "cow":
            yield Action.LOOK_LEFT
        yield Action.DO

    def wander_around(self, obs):
        # find a random direction to move
        return random.choice(
            [Action.LOOK_UP, Action.LOOK_DOWN, Action.LOOK_LEFT, Action.LOOK_RIGHT]
        )

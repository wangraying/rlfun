import numpy as np
from enum import Enum
from collections import namedtuple, defaultdict
import random
import plotly.graph_objects as go

maze1 = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]


class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


State = namedtuple(
    "State",
    ["x", "y"],
)


class DynaMaze:
    M = 6
    N = 9

    def __init__(self, maze, start: State):
        assert len(maze) == DynaMaze.M
        assert len(maze[0]) == DynaMaze.N
        self.maze = maze
        self.start = start
        self.goal = State(0, DynaMaze.N - 1)

    def set_maze(self, maze):
        self.maze = maze

    def step(self, state: State, action: Action):
        offset_x, offset_y = action.value
        x = min(max(0, state.x + offset_x), DynaMaze.M - 1)
        y = min(max(0, state.y + offset_y), DynaMaze.N - 1)

        next_state = State(x=x, y=y)
        if self.is_blocked(next_state):
            next_state = state

        r = 1 if next_state == self.goal else 0
        return next_state, r

    def is_blocked(self, state: State):
        return self.maze[state.x][state.y] == 1

    def is_goal(self, state: State):
        print(f"goal reached")
        return state == self.goal


class DynaQ:
    def __init__(
        self,
        state_space,
        action_space,
        eps=0.1,
        alpha=0.1,
        gamma=0.95,
        n_planning=0,
        seed=47,
    ):
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.n_planning = n_planning
        self.rng = np.random.default_rng(seed)

        self.action_values = defaultdict()
        for s in state_space:
            for a in action_space:
                self.action_values.setdefault(s, defaultdict())[a] = 0.0

        self.state_space = state_space
        self.action_space = action_space
        self.model = defaultdict()

    def show(self):
        for s in self.state_space:
            print(
                f"state={s}, values={[self.action_values[s][a] for a in self.action_space]}"
            )

    def get_action(self, state):
        p = self.rng.random()
        is_optimal = False
        if p < self.eps:
            idx = self.rng.choice(len(self.action_space))
        else:
            vals = np.array(
                [self.action_values[state][action] for action in self.action_space]
            )
            indices = np.argwhere(vals == vals.max()).reshape(-1)
            idx = indices[self.rng.choice(indices.shape[0])]
            is_optimal = True

        return self.action_space[idx], is_optimal

    def optimal_action(self, state: State) -> Action:
        vals = [self.action_values[state][action] for action in self.action_space]
        idx = np.argmax(vals)
        return self.action_space[idx]

    def q_learning(self, state, action, next_state, reward: int):
        td_error = (
            reward
            + self.gamma
            * max([self.action_values[next_state][a] for a in self.action_space])
            - self.action_values[state][action]
        )
        self.action_values[state][action] += self.alpha * td_error

    def update(self, state, action, next_state, reward: int):
        self.q_learning(state, action, next_state, reward)
        self.model[(state, action)] = (reward, next_state)

        for _ in range(self.n_planning):
            (s, a), (r, ns) = random.choice(list(self.model.items()))
            self.q_learning(s, a, next_state=ns, reward=r)


def optimal_path(maze: DynaMaze, agent: DynaQ):
    state = maze.start
    path = [state]
    while state != maze.goal:
        action = agent.optimal_action(state)
        print(
            f"state: {state}, take action: {action}, vals={[agent.action_values[state][a] for a in agent.action_space]}"
        )

        state, _ = maze.step(state, action)
        path.append(state)

    return path


def gen_episode(maze: DynaMaze, agent: DynaQ):
    state = maze.start
    time_step = 0
    while state != maze.goal:
        action, is_optimal = agent.get_action(state)
        next_state, reward = maze.step(state, action)
        agent.update(state, action, next_state, reward)

        state = next_state
        time_step += 1

    return time_step


if __name__ == "__main__":
    num_experiments = 10
    num_episodes = 50
    alpha = 0.1
    gamma = 0.95
    eps = 0.1

    fig = go.Figure()
    for n in [0, 5, 50]:
        exp_result = []
        for k in range(num_experiments):
            maze = DynaMaze(maze1, start=State(x=2, y=0))
            state_space = [
                State(x, y) for x in range(DynaMaze.M) for y in range(DynaMaze.N)
            ]
            action_space = list(Action)
            agent = DynaQ(
                state_space,
                action_space,
                eps=eps,
                alpha=alpha,
                gamma=gamma,
                n_planning=n,
                seed=k,
            )

            time_steps = []
            for i in range(num_episodes):
                ts = gen_episode(maze, agent)
                time_steps.append(ts)
                print(f"episode #{i}, step={ts}")

            path = optimal_path(maze, agent)
            print("Optimal path: ", path)

            exp_result.append(time_steps)
        exp_result = np.mean(exp_result, 0)

        fig.add_trace(
            go.Scatter(
                x=list(range(num_episodes))[2:],
                y=exp_result[2:],
                name=f"n_planning={n}",
            ),
        )

    fig.update_layout(
        title=f"Dyna Maze (alpha=0.1, eps=0.1, gamma=0.95)",
        showlegend=True,
        xaxis_title="Episodes",
        yaxis_title="Steps per episode",
        height=800,
        width=1200,
    )
    fig.show()

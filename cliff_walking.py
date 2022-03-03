import numpy as np
from collections import namedtuple, defaultdict
from enum import Enum
import plotly.graph_objects as go

State = namedtuple(
    "State",
    ["x", "y"],
)

M = 4
N = 12


def is_cliff(state: State):
    return state.x == M - 1 and 0 < state.y < N - 1


start_state = State(x=M - 1, y=0)
goal_state = State(x=M - 1, y=N - 1)


class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


def move(state: State, action: Action):
    offset_x, offset_y = action.value
    x = min(max(0, state.x + offset_x), 3)
    y = min(max(0, state.y + offset_y), 11)

    next_state = State(x=x, y=y)

    r = -100 if is_cliff(next_state) else -1
    ended = is_cliff(next_state) or next_state == goal_state
    return next_state, r, ended


class Agent:
    def __init__(
        self,
        state_space,
        action_space,
        eps=0.1,
        alpha=0.1,
        gamma=1.0,
        seed=47,
    ):
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.action_values = defaultdict()
        for s in state_space:
            for a in action_space:
                self.action_values.setdefault(s, defaultdict())[a] = 0.0

        self.state_space = state_space
        self.action_space = action_space

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
            idx = np.argmax(vals)
            is_optimal = True

        return self.action_space[idx], is_optimal

    def optimal_action(self, state: State) -> Action:
        vals = [self.action_values[state][action] for action in self.action_space]
        idx = np.argmax(vals)
        return self.action_space[idx]

    def update(self, state, action, next_state, reward: int, **kwargs):
        raise NotImplementedError


class Sarsa(Agent):
    def update(self, state, action, next_state, reward: int, next_action):
        td_error = (
            reward
            + self.gamma * self.action_values[next_state][next_action]
            - self.action_values[state][action]
        )
        self.action_values[state][action] += self.alpha * td_error

    def gen_episode(self):
        state = start_state
        action, _ = self.get_action(state)

        total_reward = 0
        ended = False
        while not ended:
            next_state, r, ended = move(state, action)
            next_action, _ = self.get_action(next_state)
            self.update(state, action, next_state, r, next_action)

            state = next_state
            action = next_action
            total_reward += r

        return total_reward


class QLearning(Agent):
    def update(self, state, action, next_state, reward: int, **kwargs):
        td_error = (
            reward
            + self.gamma
            * max([self.action_values[next_state][a] for a in self.action_space])
            - self.action_values[state][action]
        )
        self.action_values[state][action] += self.alpha * td_error

    def gen_episode(self):
        state = start_state
        total_reward = 0
        ended = False
        while not ended:
            action, _ = self.get_action(state)
            next_state, r, ended = move(state, action)
            self.update(state, action, next_state, r)

            state = next_state
            total_reward += r

        return total_reward


class ExpectedSarsa(QLearning):
    def update(self, state, action, next_state, reward: int, **kwargs):
        td_error = (
            reward
            + self.gamma
            * sum([self.action_values[next_state][a] for a in self.action_space])
            / len(self.action_space)
            - self.action_values[state][action]
        )
        self.action_values[state][action] += self.alpha * td_error


def optimal_path(agent: Agent):
    state = start_state
    path = [state]
    while state != goal_state:
        action = agent.optimal_action(state)
        print(
            f"state: {state}, take action: {action}, vals={[agent.action_values[state][a] for a in agent.action_space]}"
        )

        state, _, _ = move(state, action)
        path.append(state)

    return path


if __name__ == "__main__":
    num_experiments = 100
    num_episodes = 500

    state_space = [State(x, y) for x in range(M) for y in range(N)]
    action_space = list(Action)

    exp_result = []
    for k in range(num_experiments):
        sarsa = Sarsa(state_space, action_space, eps=0.1, alpha=0.1, gamma=1.0, seed=k)
        q_learning = QLearning(
            state_space, action_space, eps=0.1, alpha=0.1, gamma=1.0, seed=k
        )
        expected_sarsa = ExpectedSarsa(
            state_space, action_space, eps=0.1, alpha=1.0, gamma=1.0, seed=k
        )

        agent_rewards = []
        for agent in [sarsa, q_learning, expected_sarsa]:
            rewards = []
            for i in range(num_episodes):
                r = agent.gen_episode()
                rewards.append(r)
            agent_rewards.append(rewards)
            print(f"#{k} agent {agent} rewards: {agent_rewards}")

        exp_result.append(agent_rewards)

    exp_result = np.array(exp_result)
    exp_result = np.mean(exp_result, 0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(num_episodes))[:],
            y=exp_result[0][:],
            name="Sarsa (alpha=0.1, eps=0.1, gamma=1.0)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_episodes))[:],
            y=exp_result[1][:],
            name="Q-learning (alpha=0.1, eps=0.1, gamma=1.0)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_episodes))[:],
            y=exp_result[2][:],
            name="Expected Sarsa (alpha=1.0, eps=0.1, gamma=1.0)",
        ),
    )
    fig.update_layout(
        title=f"Cliff Walking",
        showlegend=True,
        xaxis_title="Episodes",
        yaxis_title="Sum of rewards during episode",
        height=800,
        width=1200,
    )
    fig.show()

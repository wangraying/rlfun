import numpy as np
from collections import namedtuple
from enum import Enum
import plotly.graph_objects as go

State = namedtuple(
    "State",
    ["x", "y"],
)


def is_cliff(state: State):
    return state.x == 0 and state.y > 0 and state.y < 11


# (0, 0) is the start, (0, 11) is the goal
def is_goal(state: State):
    return state.x == 0 and state.y == 11


class Action(Enum):
    UP = (1, 0)
    DOWN = (-1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


action_values = {}
action_space = list(Action)


def take_action(state: State, action: Action):
    if is_cliff(state):
        return State(0, 0), -100

    offset_x, offset_y = action.value
    x = max(0, state.x + offset_x)
    x = min(x, 3)

    y = max(0, state.y + offset_y)
    y = min(y, 11)

    next_state = State(x=x, y=y)
    r = -1
    return next_state, r


def init():
    for x in range(4):
        for y in range(12):
            s = State(x, y)
            for a in action_space:
                action_values[(s, a)] = 0.0


class SoftPolicy:
    def __init__(self, eps=0.1, seed=47):
        self.eps = eps
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: State) -> Action:
        p = self.rng.random()
        if p < self.eps:
            idx = self.rng.choice(len(action_space))
        else:
            vals = [action_values[(state, action)] for action in action_space]
            idx = np.argmax(vals)

        return action_space[idx]

    def optimal_action(self, state: State) -> Action:
        vals = [action_values[(state, action)] for action in action_space]
        idx = np.argmax(vals)
        return action_space[idx]


def optimal_path(policy):
    s = State(0, 0)
    path = [s]
    while not is_goal(s):
        action = policy.optimal_action(s)
        print(
            f"state: {s}, take action: {action}, vals={[action_values[(s, action)] for action in action_space]}"
        )

        s, _ = take_action(s, action)
        path.append(s)

    return path


def sarsa(policy, alpha=0.1, gamma=1.0):
    state = State(x=0, y=0)
    action = policy.get_action(state)

    total_reward = 0
    while not is_goal(state):
        next_state, r = take_action(state, action)
        next_action = policy.get_action(next_state)
        val = action_values[(state, action)]

        action_values[(state, action)] = val + alpha * (
            r + gamma * action_values[(next_state, next_action)] - val
        )

        state = next_state
        action = next_action
        total_reward += r

    return total_reward


def q_learning(policy, alpha=0.1, gamma=1.0):
    state = State(x=0, y=0)

    total_reward = 0
    while not is_goal(state):
        action = policy.get_action(state)
        next_state, r = take_action(state, action)
        val = action_values[(state, action)]

        action_values[(state, action)] = val + alpha * (
            r
            + gamma * max([action_values[(next_state, a)] for a in action_space])
            - val
        )

        state = next_state
        total_reward += r
    return total_reward


def expected_sarsa(policy, alpha=1.0, gamma=1.0):
    state = State(x=0, y=0)

    total_reward = 0
    while not is_goal(state):
        action = policy.get_action(state)
        next_state, r = take_action(state, action)
        val = action_values[(state, action)]

        action_values[(state, action)] = val + alpha * (
            r
            + gamma * sum([action_values[(next_state, a)] for a in action_space]) * 0.25
            - val
        )

        state = next_state
        total_reward += r
    return total_reward


def run_algorithm(algor_fn, alpha, gamma, num_episodes):
    print(algor_fn, alpha, gamma, num_episodes)
    init()
    rewards = []
    for i in range(num_episodes):
        r = algor_fn(policy, alpha=alpha, gamma=gamma)
        rewards.append(r)

    path = optimal_path(policy)
    return rewards, path


if __name__ == "__main__":

    policy = SoftPolicy(eps=0.1)
    num_episodes = 500

    # Sarsa algorithm
    sarsa_rewards, sarsa_path = run_algorithm(
        sarsa, alpha=0.1, gamma=1.0, num_episodes=num_episodes
    )
    qlearning_rewards, qlearning_path = run_algorithm(
        q_learning, alpha=0.1, gamma=1.0, num_episodes=num_episodes
    )
    exp_sarsa_rewards, exp_sarsa_path = run_algorithm(
        expected_sarsa, alpha=1.0, gamma=1.0, num_episodes=num_episodes
    )

    print("Optimal path for Sarsa: ", sarsa_path)
    print("Optimal path for Q-Learning: ", qlearning_path)
    print("Optimal path for Expected Sarsa: ", exp_sarsa_path)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(num_episodes))[10::1],
            y=sarsa_rewards[10::1],
            name="Sarsa (alpha=0.1)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_episodes))[10::1],
            y=qlearning_rewards[10::1],
            name="Q-Learning (alpha=0.1)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_episodes))[10::1],
            y=exp_sarsa_rewards[10::1],
            name="Expected Sarsa (alpha=1.0)",
        ),
    )
    fig.update_layout(
        title=f"Cliff Walking",
        showlegend=True,
        xaxis_title="Episodes",
        yaxis_title="Sum rewards during episodes",
    )
    fig.update_yaxes(range=[0, -100])
    fig.show()

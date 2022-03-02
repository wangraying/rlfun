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
    offset_x, offset_y = action.value
    x = max(0, state.x + offset_x)
    x = min(x, 3)

    y = max(0, state.y + offset_y)
    y = min(y, 11)

    next_state = State(x=x, y=y)
    r = -100 if is_cliff(next_state) else -1
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
        if is_cliff(next_state):
            # print("fall into cliff")
            # reset to start
            next_state = State(x=0, y=0)

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
            + gamma * action_values[(next_state, policy.optimal_action(next_state))]
            - val
        )

        state = next_state
        if is_cliff(state):
            # print("fall into cliff")
            # reset to start
            state = State(x=0, y=0)
        total_reward += r
    return total_reward


if __name__ == "__main__":

    policy = SoftPolicy(eps=0.1)
    num_episodes = 500

    # Sarsa algorithm
    init()
    sarsa_rewards = []
    for i in range(num_episodes):
        r = sarsa(policy, alpha=0.1, gamma=1.0)
        sarsa_rewards.append(r)

    sarsa_path = optimal_path(policy)
    print(sarsa_path)

    # Q-learning algorithm
    init()
    qlearning_rewards = []
    for i in range(num_episodes):
        r = q_learning(policy, alpha=0.1, gamma=1.0)
        qlearning_rewards.append(r)
    #
    qlearning_path = optimal_path(policy)
    print(qlearning_path)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(range(10, num_episodes)), y=sarsa_rewards[10:], name="Sarsa"),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(10, num_episodes)), y=qlearning_rewards[10:], name="Q-Learning"
        ),
    )
    fig.update_layout(
        title=f"Cliff Walking",
        showlegend=True,
        xaxis_title="Episodes",
        yaxis_title="Sum rewards during episodes",
    )
    fig.show()

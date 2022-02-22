import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Agent:
    def __init__(self, prob_win, winning_capital=100):
        self._prob_win = prob_win
        self._winning_capital = winning_capital
        self._possible_states = range(1, self._winning_capital)
        # including virtual state 0 and 100
        self._values = [0.0 for _ in range(len(self._possible_states) + 2)]
        self._values[-1] = 1.0

    def possible_actions(self, state):
        # state: 1, 2, ..., 99
        # actions: 0, 1, 2, ..., min(s, 100 - s)
        max_state = min(state, self._winning_capital - state)
        return range(max_state + 1)

    def possible_states(self):
        return self._possible_states

    def values(self):
        return self._values

    def backup_action(self, state, action, values):
        return (
            self._prob_win * values[state + action]
            + (1 - self._prob_win) * values[state - action]
        )

    def greedy_action(self, state):
        action_values = [
            self.backup_action(state, action, self.values())
            for action in self.possible_actions(state)
        ]
        # exclude action 0
        return np.argmax(action_values[1:]) + 1

    def value_update(self, state, old_values):
        self._values[state] = max(
            [
                self.backup_action(state, action, old_values)
                for action in self.possible_actions(state)
            ]
        )

    def greedy_policy(self):
        return {state: self.greedy_action(state) for state in agent.possible_states()}


def value_iteration(agent, eps=1e-12):
    delta = eps
    sweeps = 0
    ret = []

    while delta >= eps:
        old_values = agent.values().copy()
        for state in agent.possible_states():
            agent.value_update(state, old_values)
        new_values = agent.values()
        delta = np.max(np.abs(np.array(new_values) - np.array(old_values)))
        sweeps += 1

        ret.append(new_values.copy())
    return ret


if __name__ == "__main__":
    fig = go.Figure()

    probs = [
        0.4,  # 0.25, 0.55
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{}, {}]],
        subplot_titles=("Value Estimate", "Final Policy(stake)"),
    )

    selected_colors = [
        "red",
        "green",
        "yellow",
        "blue",
        "cyan",
        "purple",
        "goldenrod",
        "magenta",
    ]

    for i in range(len(probs)):
        prob_win = probs[i]
        agent = Agent(prob_win=prob_win, winning_capital=100)
        ret = value_iteration(agent, eps=1e-5)

        for sweep, value in enumerate(ret):
            fig.add_trace(
                go.Scatter(
                    x=list(agent.possible_states()),
                    y=value[1:-1],
                    name=f"sweep {sweep}",
                    line=dict(
                        color=selected_colors[sweep % len(selected_colors)],
                        width=0.5,
                    ),
                ),
                row=1,
                col=1,
            )

        final_policy = agent.greedy_policy()
        fig.add_trace(
            go.Scatter(
                x=list(final_policy.keys()),
                y=list(final_policy.values()),
                line=dict(color="firebrick", width=0.5),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Gambler's Problem", xaxis_title="Capital", yaxis_title="Value Estimates"
    )

    fig.show()

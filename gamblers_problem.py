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
        # actions: 1, 2, ..., min(s, 100 - s), exclude action 0 here for showing results better
        max_state = min(state, self._winning_capital - state)
        return range(1, max_state + 1)

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
        return np.argmax(action_values) + 1

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
        print(f"End of sweep {sweeps} delta = {delta}")
    return ret


if __name__ == "__main__":
    probs = [0.4, 0.25, 0.55]
    eps = [1e-8, 1e-5, 1e-4]
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{}, {}], [{}, {}], [{}, {}]],
        subplot_titles=(
            "prob_heads =0.4",
            "prob_heads=0.4",
            "prob_heads=0.25",
            "prob_heads=0.25",
            "prob_heads=0.55",
            "prob_heads=0.55",
        ),
    )

    for i in range(len(probs)):
        prob_win = probs[i]
        agent = Agent(prob_win=prob_win, winning_capital=100)
        ret = value_iteration(agent, eps=eps[i])

        for sweep, value in enumerate(ret):
            fig.add_trace(
                go.Scatter(
                    x=list(agent.possible_states()),
                    y=value[1:-1],
                    name=f"sweep {sweep}",
                ),
                row=i + 1,
                col=1,
            )

        final_policy = agent.greedy_policy()
        fig.add_trace(
            go.Scatter(
                x=list(final_policy.keys()),
                y=list(final_policy.values()),
                line=dict(color="firebrick"),
            ),
            row=i + 1,
            col=2,
        )

    for row in [1, 2, 3]:
        fig.update_xaxes(title_text="Capital", row=row, col=1, dtick=25)
        fig.update_xaxes(title_text="Capital", row=row, col=2, dtick=25)
        fig.update_yaxes(title_text="Value Estimates", row=row, col=1, dtick=0.2)
        fig.update_yaxes(title_text="Final Policy(stake)", row=row, col=2, dtick=10)

    fig.update_layout(title=f"Gambler's Problem", showlegend=False)

    fig.show()

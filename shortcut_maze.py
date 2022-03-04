import random
import math
import plotly.graph_objects as go
from dyna_maze import Action, State, DynaMaze, DynaQ
from blocking_maze import DynaQPlus

maze1 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

maze2 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]


class DynaQPlus(DynaQ):
    def __init__(
        self,
        state_space,
        action_space,
        eps=0.1,
        alpha=0.1,
        gamma=0.95,
        n_planning=0,
        kappa=0.01,
        seed=47,
    ):
        super(DynaQPlus, self).__init__(
            state_space,
            action_space,
            eps=eps,
            alpha=alpha,
            gamma=gamma,
            n_planning=n_planning,
            seed=seed,
        )
        self.kappa = kappa
        self.update_cnt = 0

        # initialize default model, so that those actions that have never been tried can be
        # considered in the planning step, they transfer to themselves with a reward of 0
        for state in state_space:
            for action in action_space:
                # (reward, next_state, last updated step)
                self.model[(state, action)] = (
                    0,
                    state,
                    0,
                )

    def q_learning(self, state, action, next_state, reward: int, bonus: float):
        # bonus for long-untried actions
        td_error = (
            reward
            + bonus
            + self.gamma
            * max([self.action_values[next_state][a] for a in self.action_space])
            - self.action_values[state][action]
        )
        self.action_values[state][action] += self.alpha * td_error

    def update(self, state, action, next_state, reward: int):
        self.q_learning(state, action, next_state, reward, bonus=0.0)
        self.model[(state, action)] = (reward, next_state, self.update_cnt)

        for _ in range(self.n_planning):
            (s, a), (r, ns, _) = random.choice(list(self.model.items()))

            # calculate bonus reward for simulated experiences
            _, _, last_updated = self.model[(s, a)]
            bonus = self.kappa * math.sqrt(self.update_cnt - last_updated)
            self.q_learning(s, a, next_state=ns, reward=r, bonus=bonus)

        self.update_cnt += 1

    def __repr__(self):
        return self.__class__.__name__ + f"(kappa={self.kappa})"


if __name__ == "__main__":
    num_steps = 6000
    alpha = 0.1
    gamma = 0.95
    eps = 0.1
    n_planning = 50

    blocking_maze = DynaMaze(maze1, start=State(x=DynaMaze.M - 1, y=3))
    state_space = [State(x, y) for x in range(DynaMaze.M) for y in range(DynaMaze.N)]
    action_space = list(Action)

    kwargs = {
        "alpha": alpha,
        "gamma": gamma,
        "eps": eps,
        "n_planning": n_planning,
        "seed": 47,
    }
    dyna_q = DynaQ(state_space, action_space, **kwargs)
    dyna_q_plus1 = DynaQPlus(state_space, action_space, kappa=1e-3, **kwargs)
    dyna_q_plus2 = DynaQPlus(state_space, action_space, kappa=1e-4, **kwargs)

    exp_result = []
    for agent in [dyna_q, dyna_q_plus1, dyna_q_plus2]:
        random.seed(47)
        blocking_maze.set_maze(maze1)
        step_rewards = []
        accum_reward = 0
        state = blocking_maze.start
        for i in range(num_steps):
            if state == blocking_maze.goal:
                state = blocking_maze.start

            action, _ = agent.get_action(state)
            next_state, r = blocking_maze.step(state, action)
            agent.update(state, action, next_state, r)

            state = next_state
            accum_reward += r
            step_rewards.append(accum_reward)
            print(f"{agent} step #{i}, reward={accum_reward}")
            if i == 3000:
                agent.show()
                blocking_maze.set_maze(maze2)
        exp_result.append(step_rewards)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(num_steps))[:],
            y=exp_result[0][:],
            name="Dyna-Q",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_steps))[:],
            y=exp_result[1][:],
            name=f"Dyna-Q+(kappa={dyna_q_plus1.kappa})",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_steps))[:],
            y=exp_result[2][:],
            name=f"Dyna-Q+(kappa={dyna_q_plus2.kappa})",
        ),
    )
    fig.update_layout(
        title=f"Shortcut Maze (alpha=0.1, eps=0.1, gamma=0.95, n_planning=50)",
        showlegend=True,
        xaxis_title="Time steps<br>(environment becomes easier after step 3000)",
        yaxis_title="Cumulative reward",
        height=800,
        width=1200,
    )
    fig.show()

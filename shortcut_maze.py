import random
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
    dyna_q_plus1 = DynaQPlus(state_space, action_space, kappa=1e-2, **kwargs)
    dyna_q_plus2 = DynaQPlus(state_space, action_space, kappa=1e-3, **kwargs)

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

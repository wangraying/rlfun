import plotly.graph_objects as go
from dyna_maze import Action, State, DynaMaze, DynaQ

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
    n_planning = 5

    blocking_maze = DynaMaze(maze1, start=State(x=DynaMaze.M - 1, y=3))
    state_space = [State(x, y) for x in range(DynaMaze.M) for y in range(DynaMaze.N)]
    action_space = list(Action)

    dyna_q = DynaQ(
        state_space, action_space, eps=eps, alpha=alpha, gamma=gamma, n_planning=50
    )

    step_rewards = []
    accum_reward = 0
    state = blocking_maze.start
    for i in range(num_steps):
        if state == blocking_maze.goal:
            state = blocking_maze.start

        action, _ = dyna_q.get_action(state)
        next_state, r = blocking_maze.step(state, action)
        dyna_q.update(state, action, next_state, r)

        state = next_state
        accum_reward += r
        step_rewards.append(accum_reward)
        print(f"step #{i}, reward={accum_reward}")
        if i == 3000:
            dyna_q.show()
            blocking_maze.set_maze(maze2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(num_steps))[:],
            y=step_rewards[:],
            name="Dyna-Q",
        ),
    )
    fig.update_layout(
        title=f"Shortcut Maze (alpha=0.1, eps=0.1, gamma=0.95, n_planning=50)",
        showlegend=True,
        xaxis_title="Time steps",
        yaxis_title="Cumulative reward",
        height=800,
        width=1200,
    )
    fig.show()

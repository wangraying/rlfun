import numpy as np
from collections import defaultdict
from plotly.subplots import make_subplots
from collections import namedtuple
from enum import Enum
from blackjack import Player, Dealer, AverageMeter
import argparse
import logging

logging.basicConfig(level=logging.INFO)

# deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


State = namedtuple(
    "State",
    ["usable_ace", "dealer_showing", "current_sum"],
)


class State(State):
    def __repr__(self):
        return f"({self.usable_ace}, {self.dealer_showing}, {self.current_sum})"


class Action(Enum):
    STICK = 0
    HIT = 1


state_space = [
    State(usable_ace, dealer_showing, player_sum)
    for usable_ace in [True, False]
    for dealer_showing in range(1, 11)
    for player_sum in range(12, 22)
]

action_space = list(Action)


def random_start():
    s_idx = np.random.choice(len(state_space))
    a_idx = np.random.choice(len(action_space))
    return state_space[s_idx], action_space[a_idx]


class ActionValue(dict):
    def init(self):
        # initialize values for all state-action pairs
        for state in state_space:
            for action in action_space:
                self.setdefault(state.usable_ace, defaultdict()).__setitem__(
                    (state, action), AverageMeter()
                )
        return self

    def get_avg(self, state: State, action: Action):
        return self[state.usable_ace][(state, action)].avg()

    def get_value(self, state: State, action: Action):
        return self[state.usable_ace].get((state, action), 0.0)

    def update_value(self, state: State, action: Action, r: int):
        self[state.usable_ace][(state, action)].increase(r)


class Policy(dict):
    def init(self, stick_policy=20):
        # initialize default policy for all states
        for state in state_space:
            p = self.setdefault(state.usable_ace, defaultdict())
            p[state] = Action.STICK if state.current_sum >= stick_policy else Action.HIT
        return self

    def get_action(self, state: State) -> Action:
        return self[state.usable_ace][state]

    def greedify(self, action_values: ActionValue, state: State) -> bool:
        action = self.get_action(state)

        improved = False
        for a in action_space:
            if action_values.get_avg(state, a) > action_values.get_avg(state, action):
                action = a
                improved = True
        self[state.usable_ace][state] = action
        return improved


class SoftPolicy(Policy):
    def init(self, stick_policy=20, eps=0.01, seed=47):
        super().init(stick_policy)
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        return self

    def get_action(self, state: State) -> Action:
        p = self.rng.random()
        if p < self.eps:
            idx = self.rng.choice(len(action_space))
            return action_space[idx]
        else:
            return self[state.usable_ace].get(state, Action.HIT)


# episode with exploring starts
def gen_episode_with_es(player: Player, dealer: Dealer, policy: Policy):
    state, action = random_start()

    player.init(num_ace=int(state.usable_ace), current_sum=state.current_sum)
    dealer.draw_card(state.dealer_showing)
    dealer.draw_card()

    state_actions = [(state, action)]

    # End of episode when player has a natural
    # if player.is_natural():
    #     logging.debug("player has a natural")
    #     # player wins unless dealer has natural
    #     return state_actions, (0 if dealer.is_natural() else 1)

    while action != Action.STICK:
        player.draw_card()

        # End of episode when player goes bust (player loses)
        if player.is_bust():
            logging.debug(
                f"player goes bust, {player._hands}, sum={player.current_sum()}"
            )
            return state_actions, -1

        state = State(
            usable_ace=player.usable_ace(),
            dealer_showing=dealer.showing_card(),
            current_sum=player.current_sum(),
        )
        action = policy.get_action(state)
        state_actions.append((state, action))

    # Dealer's turn when player sticks
    while not dealer.is_to_stick():
        dealer.draw_card()

        # End of episode when dealer goes bust
        if dealer.is_bust():
            logging.debug(
                f"dealer goes bust, {dealer._hands}, sum={dealer.current_sum()}"
            )
            return state_actions, 1

    if player.current_sum() == dealer.current_sum():
        return state_actions, 0

    return state_actions, (1 if player.current_sum() > dealer.current_sum() else -1)


def gen_episode(player: Player, dealer: Dealer, policy: SoftPolicy):
    for _ in range(2):
        dealer.draw_card()
        player.draw_card()

    # End of episode when player has a natural
    if player.is_natural():
        logging.debug("player has a natural")
        # player wins unless dealer has natural
        return [], (0 if dealer.is_natural() else 1)

    state = State(
        usable_ace=player.usable_ace(),
        dealer_showing=dealer.showing_card(),
        current_sum=player.current_sum(),
    )
    action = policy.get_action(state)
    state_actions = [(state, action)]

    while action != Action.STICK:
        player.draw_card()

        # End of episode when player goes bust (player loses)
        if player.is_bust():
            logging.debug(
                f"player goes bust, {player._hands}, sum={player.current_sum()}"
            )
            return state_actions, -1

        state = State(
            usable_ace=player.usable_ace(),
            dealer_showing=dealer.showing_card(),
            current_sum=player.current_sum(),
        )
        action = policy.get_action(state)
        state_actions.append((state, action))

    # Dealer's turn when player sticks
    while not dealer.is_to_stick():
        dealer.draw_card()

        # End of episode when dealer goes bust
        if dealer.is_bust():
            logging.debug(
                f"dealer goes bust, {dealer._hands}, sum={dealer.current_sum()}"
            )
            return state_actions, 1

    if player.current_sum() == dealer.current_sum():
        return state_actions, 0

    return state_actions, (1 if player.current_sum() > dealer.current_sum() else -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solving Blackjack Game")
    parser.add_argument("--num_episodes", type=int, default=5000001)
    parser.add_argument(
        "--algorithm", type=str, default="MC-ES", help="MC-ES or MC-Control"
    )
    parser.add_argument("--eps", type=float, default=0.01)

    args = parser.parse_args()
    if args.algorithm == "MC-ES":
        policy = Policy().init(stick_policy=20)
        gen_episode_fn = gen_episode_with_es
        title_text = "Solving Blackjack with Monte Carlo Exploring Starts"
    else:
        policy = SoftPolicy().init(stick_policy=20, eps=args.eps)
        gen_episode_fn = gen_episode
        title_text = "Solving Blackjack with Monte Carlo Control (eps={})".format(
            args.eps
        )

    last_improved = 0
    action_values = ActionValue().init()

    for i in range(args.num_episodes):
        if i - last_improved > 200000:
            logging.info(f"Converged at episode #{i}")
            break

        player = Player()
        dealer = Dealer(stick_policy=17)
        state_actions, G = gen_episode_fn(player, dealer, policy)
        verbose = True

        for t, state_action in reversed(list(enumerate(state_actions))):
            s, a = state_action

            # only consider player_sum in [12, 21]
            if s.current_sum < 12 or s.current_sum > 21:
                continue

            # First-visit method
            if (s, a) not in state_actions[:t]:
                action_values.update_value(state=s, action=a, r=G)

                if verbose:
                    logging.debug(f"episode #{i}: t={t} sa=({s}, {a}),  G={G}")

                if policy.greedify(action_values, s):
                    last_improved = i
                    logging.info(
                        f"episode #{i}: policy improved: state={s}, value={[action_values.get_avg(s, _a) for _a in action_space]}"
                    )

        if verbose:
            logging.debug(
                f"After episode #{i}: dealer_hands={dealer._hands}, dealer_sum={dealer.current_sum()}, players hands={player._hands}, player_sum={player.current_sum()}, G={G}"
            )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"is_3d": True}, {"is_3d": True}], [{}, {}]],
        subplot_titles=(
            "Optimal Value (No Usable Ace)",
            "Optimal Value (Usable Ace)",
            "Optimal Policy (No Usable Ace)<br>1: hit, 0: stick",
            "Optimal Policy (Usable Ace)<br>1: hit, 0: stick",
        ),
    )

    for col in [1, 2]:
        usable_ace = bool(col - 1)
        if usable_ace not in action_values:
            continue

        dealer_showings = np.linspace(1, 10, num=10, dtype=int)
        player_sums = np.linspace(12, 21, num=10, dtype=int)

        def get_estimate(ds, ps):
            values = [
                action_values.get_avg(
                    State(usable_ace=usable_ace, dealer_showing=ds, current_sum=ps),
                    action,
                )
                for action in action_space
            ]
            val = max(values)
            return val

        estimates = [
            [get_estimate(ds, ps) for ds in dealer_showings] for ps in player_sums
        ]

        # Define the first family of coordinate lines
        line_marker = dict(color="#101010", width=4)
        dealer_showings_grid, player_sums_grid = np.meshgrid(
            dealer_showings, player_sums
        )

        for player_sums_exp, dealer_showings_exp, estimates_exp in zip(
            player_sums_grid, dealer_showings_grid, estimates
        ):
            fig.add_scatter3d(
                x=player_sums_exp,
                y=dealer_showings_exp,
                z=estimates_exp,
                mode="lines",
                line=line_marker,
                name="",
                row=1,
                col=col,
            )

        # Define the second family of coordinate lines
        estimates = np.transpose(estimates)
        player_sums_grid, dealer_showings_grid = np.meshgrid(
            player_sums, dealer_showings
        )

        for player_sums_exp, dealer_showings_exp, estimates_exp in zip(
            player_sums_grid, dealer_showings_grid, estimates
        ):
            fig.add_scatter3d(
                x=player_sums_exp,
                y=dealer_showings_exp,
                z=estimates_exp,
                mode="lines",
                line=line_marker,
                name="",
                row=1,
                col=col,
            )

        # add figure for optimal policy
        optimal_policy = [
            [
                policy.get_action(
                    State(usable_ace=usable_ace, dealer_showing=ds, current_sum=ps)
                ).value
                for ds in dealer_showings
            ]
            for ps in player_sums
        ]
        fig.add_heatmap(
            x=dealer_showings,
            y=player_sums,
            z=optimal_policy,
            row=2,
            col=col,
            showscale=False,
            colorscale="Mint",
            opacity=0.5,
        )

    fig.update_xaxes(title_text="Dealer showing", row=2, col=1, dtick=1)
    fig.update_xaxes(title_text="Dealer showing", row=2, col=2, dtick=1)
    fig.update_yaxes(title_text="Player sum", row=2, col=1, dtick=1)
    fig.update_yaxes(title_text="Player sum", row=2, col=2, dtick=1)
    fig.update_layout(
        title_text=title_text,
        scene=dict(
            xaxis_title="Player sum",
            yaxis_title="Dealer showing",
            zaxis_title="",
            xaxis=dict(
                autorange="reversed",
            ),
        ),
        scene2=dict(
            xaxis_title="Player sum",
            yaxis_title="Dealer showing",
            zaxis_title="",
            xaxis=dict(
                autorange="reversed",
            ),
        ),
        height=1200,
        width=1200,
        showlegend=False,
    )
    fig.show()

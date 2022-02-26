import numpy as np
from collections import defaultdict
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)

# deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def draw_card():
    # All face cards count as 10
    return min(np.random.choice(deck), 10)


def is_ace(card):
    return card == 1


class Player:
    def __init__(self, stick_policy=20):
        self._stick_policy = stick_policy
        self._current_sum = 0
        self._num_ace = 0
        self._hands = []

    def is_to_stick(self):
        return self.current_sum() >= self._stick_policy

    def hit(self):
        c = draw_card()
        self._current_sum += c

        if is_ace(c):
            self._num_ace += 1
            if self._current_sum + 10 <= 21:
                self._current_sum += 10
        self._hands.append(c)
        return c

    def is_bust(self):
        return self.current_sum() > 21

    def is_natrual(self):
        return self.current_sum() == 21 and self._num_ace == 1

    def current_sum(self):
        return self._current_sum


class Dealer(Player):
    def __init__(self, stick_policy=17):
        super(Dealer, self).__init__(stick_policy)
        self._showing_card = None

    def hit(self):
        c = super().hit()
        if self._showing_card is None:
            self._showing_card = c

    def showing_card(self):
        return self._showing_card


def gen_episode(player, dealer):
    for _ in range(2):
        dealer.hit()
        player.hit()

    # End of episode when player has a natural
    if player.is_natrual():
        logging.debug("player has a natural")
        # player wins unless dealer has natural
        return 0 if dealer.is_natrual() else 1

    while not player.is_to_stick():
        player.hit()

        # End of episode when player goes bust (player loses)
        if player.is_bust():
            logging.debug(f"player goes bust, {player._hands}")
            return -1

    # Dealer's turn when player sticks
    while not dealer.is_to_stick():
        dealer.hit()

        # End of episode when dealer goes bust
        if dealer.is_bust():
            logging.debug(f"dealer goes bust, {dealer._hands}")
            return 1

    if player.current_sum() == dealer.current_sum():
        return 0

    return 1 if player.current_sum() > dealer.current_sum() else -1


def recover_states(player):
    sum = 0
    usable_ace = False
    states = []

    for c in player._hands:
        sum += c
        if is_ace(c):
            if sum + 10 <= 21:
                usable_ace = True
                sum += 10

        states.append((usable_ace and sum <= 21, sum))
    return states


class AverageMeter:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def increase(self, n):
        self._sum += n
        self._count += 1

    def avg(self):
        return self._sum / self._count

    def __repr__(self):
        return f"(sum={self._sum}, count={self._count})"


if __name__ == "__main__":
    num_episodes = 10000
    values = {}

    for i in range(num_episodes):
        player = Player()
        dealer = Dealer()
        G = gen_episode(player, dealer)
        player_states = recover_states(player)
        verbose = 1 in player._hands

        visited_states = set()
        for s in player_states:
            usable_ace, player_sum = s
            # only consider player_sum in [12, 21]
            if player_sum < 12 or player_sum > 21:
                continue

            new_s = (dealer.showing_card(), player_sum)

            if usable_ace not in values:
                values[usable_ace] = defaultdict()

            # First-visit method
            if new_s not in visited_states:
                values[usable_ace].setdefault(new_s, AverageMeter()).increase(G)

            if verbose:
                logging.debug(
                    f"episode #{i}: players hands={player._hands}, s={s}, G={G}"
                )

        if verbose:
            logging.debug(
                f"After episode #{i}: dealer_hands={dealer._hands}, dealer_sum={dealer.current_sum()}, players hands={player._hands}, player_sum={player.current_sum()}, G={G}"
            )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"is_3d": True}, {"is_3d": True}],
        ],
        subplot_titles=("No Usable Ace", "Usable Ace"),
    )

    for col in [1, 2]:
        usable_ace = bool(col - 1)
        if usable_ace not in values:
            continue

        states = values[usable_ace].keys()
        dealer_showings = np.linspace(1, 10, num=10, dtype=int)
        player_sums = np.linspace(12, 21, num=10, dtype=int)

        def get_estimate(ds, ps):
            val = values[usable_ace].get((ds, ps))
            avg_val = 0.0 if val is None else val.avg()
            logging.info(
                f"usable_ace={usable_ace} (dealer showing={ds}, player sum={ps}), value={val}, {avg_val}"
            )
            return avg_val

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

    fig.update_layout(
        title_text="Blackjack Game (After {:,} episodes)".format(num_episodes),
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
        height=1000,
        showlegend=False,
    )
    fig.show()

import numpy as np
from collections import defaultdict
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)

# deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def random_card():
    # All face cards count as 10
    return min(np.random.choice(deck), 10)


def is_ace(card):
    return card == 1


class Player:
    def __init__(self, stick_policy=20):
        self._fixed_policy = stick_policy
        self._current_sum = 0
        self._num_ace = 0
        self._hands = []

    def is_to_stick(self):
        return self.current_sum() >= self._fixed_policy

    def draw_card(self, card=None) -> int:
        if card is None:
            card = random_card()

        self._current_sum = self._current_sum + card

        if is_ace(card):
            self._num_ace += 1

        self._hands.append(card)
        return card

    def is_bust(self):
        return self.current_sum() > 21

    def is_natural(self):
        return self.current_sum() == 21 and self._num_ace == 1

    def current_sum(self):
        return self._current_sum + 10 if self.usable_ace() else self._current_sum

    def usable_ace(self):
        return self._num_ace > 0 and (self._current_sum + 10 <= 21)

    def init(self, num_ace: int, current_sum: int):
        self._current_sum = current_sum - 10 if num_ace else current_sum
        self._num_ace = num_ace
        self._hands.append(current_sum)  # for debug use


class Dealer(Player):
    def __init__(self, stick_policy=17):
        super(Dealer, self).__init__(stick_policy)
        self._showing_card = None

    def draw_card(self, card=None) -> int:
        card = super().draw_card(card)
        if self._showing_card is None:
            self._showing_card = card
        return card

    def showing_card(self):
        return self._showing_card


def gen_episode(player, dealer):
    for _ in range(2):
        dealer.draw_card()
        player.draw_card()

    states = [(player.usable_ace(), player.current_sum())]
    # End of episode when player has a natural
    if player.is_natural():
        logging.debug("player has a natural")
        # player wins unless dealer has natural
        return states, (0 if dealer.is_natural() else 1)

    while not player.is_to_stick():
        player.draw_card()

        # End of episode when player goes bust (player loses)
        if player.is_bust():
            logging.debug(f"player goes bust, {player._hands}")
            return states, -1

        states.append((player.usable_ace(), player.current_sum()))

    # Dealer's turn when player sticks
    while not dealer.is_to_stick():
        dealer.draw_card()

        # End of episode when dealer goes bust
        if dealer.is_bust():
            logging.debug(f"dealer goes bust, {dealer._hands}")
            return states, 1

    if player.current_sum() == dealer.current_sum():
        return states, 0

    return states, (1 if player.current_sum() > dealer.current_sum() else -1)


class AverageMeter:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def increase(self, n):
        self._sum += n
        self._count += 1

    def avg(self):
        return self._sum / self._count if self._count != 0 else 0

    def __repr__(self):
        return f"(sum={self._sum}, count={self._count})"


if __name__ == "__main__":
    num_episodes = 10000
    values = {}

    for i in range(num_episodes):
        player = Player()
        dealer = Dealer()
        player_states, G = gen_episode(player, dealer)
        verbose = 1 in player._hands

        for t, s in reversed(list(enumerate(player_states))):
            usable_ace, player_sum = s
            # only consider player_sum in [12, 21]
            if player_sum < 12 or player_sum > 21:
                continue

            # First-visit method
            if s not in player_states[:t]:
                if usable_ace not in values:
                    values[usable_ace] = defaultdict()

                new_s = (dealer.showing_card(), player_sum)
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
            val = values[usable_ace].get((ds, ps)).avg()
            logging.info(
                f"usable_ace={usable_ace} (dealer showing={ds}, player sum={ps}), value={val}"
            )
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

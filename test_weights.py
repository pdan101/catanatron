from catanatron import Game, RandomPlayer, Color, RandomPlayer

from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron_experimental.play import play_batch

from catanatron_experimental.machine_learning.players.value import DEFAULT_WEIGHTS

test_weights = DEFAULT_WEIGHTS.copy()
for k, v in test_weights.items():
    test_weights[k] = 0
test_weights['enemy_production'] = -5e8


players = [
    # AlphaBetaPlayer(Color.RED, params=test_weights, value_fn_builder_name="A"),
    RandomPlayer(Color.BLUE),
    AlphaBetaPlayer(Color.RED)
    # MCTSPlayer(Color.BLUE)
]

wins, results_by_player, games = play_batch(10, players)



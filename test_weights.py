from catanatron import Game, RandomPlayer, Color, RandomPlayer

from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron_experimental.play import play_batch

from catanatron_experimental.machine_learning.players.value import DEFAULT_WEIGHTS

from itertools import product
import pandas as pd

NUM_GAMES_EACH_SETTING = 100
######## DataFrame for output data ##############################
player_summary = pd.DataFrame()
game_summary = pd.DataFrame()
######## Define the weights we are interested in ################
def set_equal_weights(weights, val):
    for k, _ in weights.items():
        weights[k] = 0
    return weights

default_w_0 = set_equal_weights(DEFAULT_WEIGHTS.copy(), 0)
AlphaBetaFullAd_w = default_w_0.copy()
AlphaBetaFullAd_w['enemy_production'] = -5e8

AlphaBetaFullPr_w = default_w_0.copy()
AlphaBetaFullPr_w['production'] = 5e8

default_w_1 = set_equal_weights(DEFAULT_WEIGHTS.copy(), 1e3)
AlphaBetaMixedAd_w = default_w_1.copy()
AlphaBetaMixedAd_w['enemy_production'] = -5e8

AlphaBetaMixedPr_w = default_w_1.copy()
AlphaBetaMixedAd_w['production'] = 5e8

######### Define the player we are interested in #################
AlphaBetaFullAd = AlphaBetaPlayer(Color.RED)
AlphaBetaMixedAd = AlphaBetaPlayer(Color.RED)
AlphaBetaFullPr = AlphaBetaPlayer(Color.RED)
AlphaBetaMixedPr = AlphaBetaPlayer(Color.RED)

new_agents = {AlphaBetaFullAd: 'abFullAd', AlphaBetaMixedAd: 'abMixedAd', AlphaBetaFullPr: 'abFullPr', AlphaBetaMixedPr: 'abMixedPr'} 
######### Define the baseline players ############################
AlphaBeta = AlphaBetaPlayer(Color.ORANGE)
mcts = MCTSPlayer(Color.BLUE)
Random = RandomPlayer(Color.WHITE)

base_agents = {AlphaBeta: 'ab', mcts: 'mcts', Random: 'random'}

######## Define the set of players for each game ################
num_comp = len(base_agents.keys()) * len(new_agents.keys())
comp = list(product(new_agents.keys(), base_agents.keys()))

######## Play games #############################################
for idx, (new_agent, base_agent) in enumerate(comp):
    players = [new_agent, base_agent]
    # I modified catanatron/catanatron_experimental/catanatron_experimental/play.py to make it  take in more argument for documentation purposes
    _, _, _, df_g, df_p = play_batch(NUM_GAMES_EACH_SETTING, players, idx = idx, player_names = [new_agents[new_agent], base_agents[base_agent]])
    game_summary = pd.concat([game_summary, df_g], axis = 0)
    player_summary = pd.concat([player_summary, df_p], axis = 0)

# ######## Tiny Example #########################################
# test_weights = DEFAULT_WEIGHTS.copy()
# for k, v in test_weights.items():
#     test_weights[k] = 1e3
# # test_weights['enemy_production'] = -5e8
# test_weights['enemy_production'] = -5e8

# players = [
#     AlphaBetaPlayer(Color.RED, params=test_weights, value_fn_builder_name="A"),
#     AlphaBetaPlayer(Color.BLUE),
# ]

# wins, results_by_player, games, df_p, df_g = play_batch(1, players, idx = 0, player_names = ['small_red', 'small_blue'])

# game_summary = pd.concat([game_summary, df_g], axis = 0)
# player_summary = pd.concat([player_summary, df_p], axis = 0)

game_summary.to_csv("/Users/sl984/Documents/Fall23/ORIE4350/catan/catanatron/results/game_test.csv")
player_summary.to_csv("/Users/sl984/Documents/Fall23/ORIE4350/catan/catanatron/results/player_test.csv")




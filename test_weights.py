from catanatron import Game, RandomPlayer, Color, RandomPlayer

from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron_experimental.play import play_batch

from catanatron_experimental.machine_learning.players.value import DEFAULT_WEIGHTS

from itertools import product
import pandas as pd

NUM_GAMES_EACH_SETTING = 200
######## DataFrame for output data ##############################
player_summary = pd.DataFrame()
game_summary = pd.DataFrame()
######## Define the weights we are interested in ################
def set_equal_weights(weights, val):
    for k, _ in weights.items():
        weights[k] = 0
    return weights

default_w_0 = set_equal_weights(DEFAULT_WEIGHTS.copy(), 0)
### Fully Adversarial - all weights 0 except enemy production
AlphaBetaFullAd_w = default_w_0.copy()
AlphaBetaFullAd_w['enemy_production'] = -5e12

default_w_1 = DEFAULT_WEIGHTS.copy()
### Fully Production - all weights same as default but 0 enemy production
AlphaBetaFullPr_w = default_w_1.copy()
AlphaBetaFullPr_w['enemy_production'] = 0

### Mostly Adversarial, Slightly Production - divide all weights by 3 and increase
###                                           importance of enemy production
AlphaBetaMixedAd_w = default_w_1.copy()
for k, v in AlphaBetaMixedAd_w.items():
    AlphaBetaMixedAd_w[k] = v/3
AlphaBetaMixedAd_w['enemy_production'] = -5e12

### Mostly Production, Slightly Adversarial - default weights but enemy production 
###                                           only slightly negative
AlphaBetaMixedPr_w = default_w_1.copy()
AlphaBetaMixedPr_w['enemy_production'] = -1e4

######### Define the player we are interested in #################
AlphaBetaFullAd = AlphaBetaPlayer(Color.RED, params=AlphaBetaFullAd_w, value_fn_builder_name="A")
AlphaBetaMixedAd = AlphaBetaPlayer(Color.RED, params=AlphaBetaMixedAd_w, value_fn_builder_name="A")
AlphaBetaFullPr = AlphaBetaPlayer(Color.RED, params=AlphaBetaFullPr_w, value_fn_builder_name="A")
AlphaBetaMixedPr = AlphaBetaPlayer(Color.RED, params=AlphaBetaMixedPr_w, value_fn_builder_name="A")

# new_agents = {AlphaBetaFullAd: 'abFullAd', AlphaBetaMixedAd: 'abMixedAd', AlphaBetaFullPr: 'abFullPr', AlphaBetaMixedPr: 'abMixedPr'} 
new_agents = {AlphaBetaMixedAd: 'abMixedAd', AlphaBetaFullPr: 'abFullPr'} 
######### Define the baseline players ############################
AlphaBeta = AlphaBetaPlayer(Color.ORANGE)
mcts = MCTSPlayer(Color.BLUE)
Random = RandomPlayer(Color.WHITE)

base_agents = {AlphaBeta: 'ab', mcts: 'mcts', Random: 'random'}
# base_agents = {Random: 'random'}

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
import time 
timestr = time.strftime("%Y%m%d-%H%M%S")
game_summary.to_csv(f"./results/game_test_{timestr}.csv")
player_summary.to_csv(f"./results/player_test_{timestr}.csv")




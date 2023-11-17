from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player
import random

@register_player("RANDOM")
class RandomPlayer(Player):
  def decide(self, game, playable_actions):
    """Should return one of the playable_actions.

    Args:
        game (Game): complete game state. read-only.
        playable_actions (Iterable[Action]): options to choose from
    Return:
        action (Action): Chosen element of playable_actions
    """
    # ===== YOUR CODE HERE =====
    # As an example we simply return the first action:
    action_idx = random.randint(0, len(playable_actions) - 1)
    return playable_actions[action_idx]
    # ===== END YOUR CODE =====
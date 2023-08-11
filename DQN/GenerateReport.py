import os
import matplotlib.pyplot as plt

import numpy as np

# usefull functions
def mov_avg(x, n):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def mov_std(x, n):
    windows = [x[i:i+n] for i in range(len(x)-n+1)]
    return np.array([np.std(w) for w in windows])

def generate_summary(scores,
                     losses,
                     wins,
                     test_wins,
                     nr_test_episodes,
                     transitions,
                     path):
    """
    Makes plots that summarize the training process.

    Args:
        scores (list): list of scores
        losses (list): list of losses
        wins (list): list of wins (1 if won, 0 if draw, -1 if lost)
        test_wins (list): list of wins in test games 
        nr_test_episodes (int): number of test games
        path (str): path to save the plots
    """
    # numpy conversion
    scores = np.array(scores)
    losses = np.array(losses)
    wins = np.array(wins)
    test_wins = np.array(test_wins)
    transitions = np.array(transitions)
    
    # scores
    plt.plot(scores, ".",
             markersize=1,
             color="tab:blue",
             alpha=0.5,
             label="individual scores")
    plt.plot(np.arange(len(scores) - 99) + 100,
             mov_avg(scores, 100), 
             label="100 episode moving average",
             color="tab:orange",
             linewidth=3,)
    plt.title("Scores During Training")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.xlim(100, len(scores))
    plt.grid()
    plt.legend()
    plt.savefig(path + "scores.pdf")
    plt.close()

    # train win rates
    plt_wins = np.copy(wins)
    plt_wins[wins == 0] = -1
    plt.plot(np.arange(len(plt_wins) - 99) + 100,
             mov_avg((plt_wins + 1) / 2, 100),
             label="100 episode moving average",
             color="tab:orange",
             linewidth=3,)
    plt.title("Win-Rate During Training")
    plt.xlabel("Episode")
    plt.ylabel("win-rate")
    plt.grid()
    plt.legend()
    plt.xlim(100, len(wins))
    plt.savefig(path + "training_win_rate.pdf")
    plt.close()

    # test wins
    test_wins = test_wins.reshape(-1, nr_test_episodes)
    n_lost = [sum(test_wins[i] == -1) for i in range(len(test_wins))]
    n_draw = [sum(test_wins[i] == 0) for i in range(len(test_wins))]
    n_lost = np.array(n_lost)
    n_draw = np.array(n_draw)
    
    plt.bar(np.arange(len(test_wins)) + 1,
            1 - (n_lost + n_draw)/nr_test_episodes,
            bottom=(n_lost + n_draw)/nr_test_episodes,
            color="tab:green",
            label="won",
            width=1.05)
    
    plt.bar(np.arange(len(test_wins)) + 1,
            (n_lost + n_draw)/nr_test_episodes - 
                n_lost/nr_test_episodes,
            bottom=n_lost/nr_test_episodes,
            color="tab:grey",
            label="draw",
            width=1.05)

    plt.bar(np.arange(len(test_wins)) + 1,
            n_lost/nr_test_episodes,
            color="tab:red",
            label="lost",
            width=1.05)

    plt.xlim([0.5, len(test_wins) + 0.5])
    plt.ylim([0, 1])
    
    plt.title("Test Game Winns")
    plt.xlabel(f"checkpoint")
    plt.ylabel(f"win shares in test episodes")
    plt.legend()
    plt.savefig(path + "test_wins.pdf")
    plt.close()

    # losses
    plt.plot(losses, ".", markersize=1)
    plt.plot(np.arange(len(losses) - 999) + 1000,
             mov_avg(losses, 1000),
             color="tab:orange",
             linewidth=3,
             label="1000 step moving averge")
    plt.title("Loss over Opimization Steps")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid()
    plt.xlim(0, len(losses))
    plt.savefig(path + "losses.pdf")
    plt.close()

    # transitions
    plt.plot(transitions, ".", markersize=1, label="individual transitions")
    plt.plot(np.arange(len(transitions) - 99) + 100,
             mov_avg(transitions, 100),
             label="100 episode moving average",
             color="tab:orange",
             linewidth=3,)
    plt.title("Training Episode Length")
    plt.xlabel("episode")
    plt.ylabel("transitions")
    plt.grid()
    plt.xlim(0, len(transitions))
    plt.legend()
    plt.savefig(path + "transitions.pdf")
    plt.close()


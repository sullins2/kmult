# !pip install pycapnp

from game import Game
from cfr import Cfr
from komwu import Komwu
from m2wu import m2wu

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=10000)

import argparse
import logging
import json

parser = argparse.ArgumentParser()
# parser.add_argument("game", type=str, help="Game file")
parser.add_argument("--algo", type=str, choices=['komwu', 'cfr_rm', 'cfr_rm+'], help="Learning algorithm",
                    default='komwu')
parser.add_argument("--eta", type=float, help="Learning rate (only for KOMWU)", default=0.1)
parser.add_argument("--T", type=int, help="Number of iterations", default=100)
parser.add_argument("--json", type=str, help="Output JSON file")

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s|>%(levelname)s] %(message)s')

if __name__ == '__main__':
    args = parser.parse_args()

    TOTAL = 2000
    regret_KOMWU = []
    regret_KOMWU_b = []
    regret_FLBR = []
    regret_m2wu = []

    avg_pol_KOMWU = 0.0
    avg_pol_KOMWU_b = 0.0
    avg_pol_FLBR = 0.0
    ITS = [n for n in range(TOTAL)]

    b_track = [[],[]]

    recent_strategies = []

    # Set to true to do [KOMWU, KOMWU_b, FLBR, M2WU]
    experiments = [True, True, False, False]

    # Set to true for policy, false for regret
    policy = False
    # If policy = True, do average policy or current
    do_avg_policy = False
    policy_player = 0 # player to use for policy
    policy_index = 0 # which sequence to show

    # game = Game(args.game)
    game = Game("game_instances\L23.game")
    # logging.info(f"=== Parsed game {args.game}")
    logging.info(f"    Num players:   {game.n_players}")
    logging.info(f"    Num infosets:  {sum([tpx.n_infosets for tpx in game.tpxs])}")
    logging.info(f"    Num sequences: {sum([tpx.n_sequences for tpx in game.tpxs])}")
    logging.info(f"    Payoff matrix: {game.payoff_matrix_nnz} nnz")
    logging.info(f"===========================")

    def make_agent(player):
        args.eta = 2
        if args.algo == "komwu":
            print("DOING KOMWU")
            return Komwu(game.tpxs[player], args.eta, player, False)
        elif args.algo == "komwu_b":
            print("DOING KOMWU_b")
            return Komwu(game.tpxs[player], args.eta, player, True)
        elif args.algo == "cfr_rm":
            return Cfr(game.tpxs[player], plus=False)
        elif args.algo == "cfr_rm+":
            return Cfr(game.tpxs[player], plus=True)
        elif args.algo == "m2wu":
            return m2wu(game.tpxs[player], eta=0.5, mu=0.1, N=100)



    ################################################
    # KOMWU
    ################################################
    print("Doing KOMWU original now")
    args.algo = "komwu"
    agents = [make_agent(player) for player in range(game.n_players)]

    dps = []
    args.T = TOTAL if experiments[0] else 0
    for t in range(1, args.T + 1):
        strategies = {}

        for player in range(game.n_players):
            strategies[player] = agents[player].next_strategy()
            # print("STRAT FOR p=",player, ": ", strategies[player])

        # UNIFORM SF STRATEGIES
        # if t == 1:
        #     strategies[0] = [0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 1.0]
        #     strategies[1] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]

        for player in range(game.n_players):
            gradient = game.utility_gradient(player, strategies)
            agents[player].observe_gradient(gradient,t, first_step=False, komwu=True)
        b_track[0].append(agents[policy_player].b[policy_index])

        if policy:
            if do_avg_policy:
                avg_pol_KOMWU += agents[policy_player].next_strategy()[policy_index]
                regret_KOMWU.append(avg_pol_KOMWU / t)
            else:
                regret_KOMWU.append(agents[policy_player].next_strategy()[policy_index])

        else:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            regret_KOMWU.append(max_regret)


        if t % 1000 == 0:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            logging.info(f"Iteration {t:5}  regrets  {regrets}   max_regret {max(regrets)}")

    print(agents[0].b)
        # dps.append({'iteration': t, 'regrets': regrets})

    # if args.json:
    #     with open(args.json, 'w') as outfile:
    #         json.dump(dps, outfile)

    # print(strategies)

    ################################################
    # KOMWU_b
    ################################################
    print("Doing KOMWU_b original now")
    args.algo = "komwu_b"
    agents = [make_agent(player) for player in range(game.n_players)]

    dps = []
    args.T = TOTAL if experiments[1] else 0
    for t in range(1, args.T + 1):
        strategies = {}

        for player in range(game.n_players):
            strategies[player] = agents[player].next_strategy()
            # print("STRAT FOR p=",player, ": ", strategies[player])

        # UNIFORM SF STRATEGIES
        # if t == 1:
        #     strategies[0] = [0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 1.0]
        #     strategies[1] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]

        for player in range(game.n_players):
            gradient = game.utility_gradient(player, strategies)
            agents[player].observe_gradient(gradient, t, first_step=False, komwu=True)
        b_track[1].append(agents[policy_player].b[policy_index])

        if policy:
            if do_avg_policy:
                avg_pol_KOMWU_b += agents[policy_player].next_strategy()[policy_index]
                regret_KOMWU_b.append(avg_pol_KOMWU / t)
            else:
                regret_KOMWU_b.append(agents[policy_player].next_strategy()[policy_index])

        else:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            regret_KOMWU_b.append(max_regret)

        if t % 1000 == 0:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            logging.info(f"Iteration {t:5}  regrets  {regrets}   max_regret {max(regrets)}")

        # dps.append({'iteration': t, 'regrets': regrets})
    print(agents[0].b)
    # if args.json:
    #     with open(args.json, 'w') as outfile:
    #         json.dump(dps, outfile)

    # print(strategies)

    ################################################
    # FLBR
    ################################################
    print("Doing FLBR now")
    agents = [make_agent(player) for player in range(game.n_players)]
    args.algo = "komwu"
    dps = []
    args.T = TOTAL if experiments[2] else 0
    for t in range(1, args.T + 1):
        strategies = {}

        # Get first normal x's
        for player in range(game.n_players):
            strategies[player] = agents[player].next_strategy()

        # Compute x^hats (stored in x_xi)
        for player in range(game.n_players):
            gradient = game.utility_gradient(player, strategies)
            agents[player].observe_gradient(gradient,t, first_step=True, komwu=False)

        strategies = {}

        # Get x^hats
        for player in range(game.n_players):
            strategies[player] = agents[player].next_strategy_xi()

        # Compute x's
        for player in range(game.n_players):
            gradient = game.utility_gradient(player, strategies)
            agents[player].observe_gradient(gradient, t, first_step=False, komwu=False)



        if policy:
            if do_avg_policy:
                avg_pol_FLBR += agents[policy_player].next_strategy()[policy_index]
                regret_FLBR.append(avg_pol_FLBR / t)
            else:
                regret_FLBR.append(agents[policy_player].next_strategy()[policy_index])
        else:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            regret_FLBR.append(max_regret)
        if t % 1000 == 0:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            logging.info(f"Iteration {t:5}  regrets  {regrets}   max_regret {max(regrets)}")

        # dps.append({'iteration': t, 'regrets': regrets})

    ################################################
    # M2WU
    ################################################
    print("Doing M2WU now")
    args.algo = "m2wu"
    agents = [make_agent(player) for player in range(game.n_players)]

    dps = []
    args.T = TOTAL if experiments[3] else 0
    for t in range(1, args.T + 1):
        strategies = {}

        for player in range(game.n_players):
            strategies[player] = agents[player].next_strategy()

        for player in range(game.n_players):
            gradient = game.utility_gradient(player, strategies)
            agents[player].observe_gradient(gradient,t)

        # if t % 100 == 0:
        #     for player in range(game.n_players):
        #         print(agents[player].next_strategy())


        if policy:
            regret_m2wu.append(agents[policy_player].next_strategy()[policy_index])
        else:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            regret_m2wu.append(max_regret)
        if t % 1000 == 0:
            regrets = [agent.regret() for agent in agents]
            max_regret = max(regrets)
            logging.info(f"Iteration {t:5}  regrets  {regrets}   max_regret {max(regrets)}")

        # dps.append({'iteration': t, 'regrets': regrets})

    # if args.json:
    #     with open(args.json, 'w') as outfile:
    #         json.dump(dps, outfile)


    # Uncomment below for plots
    if experiments[0]:
        plt.plot(ITS, regret_KOMWU, label='KOMWU', color='red')
    if experiments[1]:
        plt.plot(ITS, regret_KOMWU_b, label='KOMWU_b', color='purple')
    if experiments[2]:
        plt.plot(ITS, regret_FLBR, label='FLBR', color='blue')
    if experiments[3]:
        plt.plot(ITS, regret_m2wu, label='M2WU', color='green')
    plt.xlabel('Iterations')
    if policy:
        plt.ylabel('Policy')
    else:
        plt.ylabel('Max Regret')
    plt.legend()
    plt.show()

    plt.plot(ITS, b_track[0], label='KOMWU', color='green')
    plt.plot(ITS, b_track[1], label='KOMWU_b', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Max Regret')
    plt.legend()
    plt.show()

    if args.json:
        with open(args.json, 'w') as outfile:
            json.dump(dps, outfile)

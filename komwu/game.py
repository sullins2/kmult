import sys
import capnp
import numpy as np
import itertools

import game_capnp

class Treeplex(object):
    class Infoset(object):
        def __init__(self, capnp_obj, infoset_id):
            self.start_sequence_id = capnp_obj.startSequenceId
            self.end_sequence_id = capnp_obj.endSequenceId
            self.parent_sequence_id = capnp_obj.parentSequenceId
            self.n_sequences = self.end_sequence_id - self.start_sequence_id + 1
            self.infoset_id = infoset_id
            # print("ID: ", infoset_id, " start: ", self.start_sequence_id, " end: ", self.end_sequence_id)

    def __init__(self, capnp_tpx):
        self.infosets = [self.Infoset(infoset, infoset_id)
                         for infoset_id, infoset in enumerate(capnp_tpx.infosets)]

        # for id, infoset in enumerate(capnp_tpx.infosets):
        #     print(id, infoset)
        self.n_infosets = len(self.infosets)
        self.n_sequences = max(
            [infoset.parent_sequence_id + 1 for infoset in self.infosets])
        self.root_sequence_id = self.n_sequences - 1

        # Build the association from parent sequence to children infosets with that
        # parent sequence
        self.children = [[] for _ in range(self.n_sequences)]
        for infoset in self.infosets:
            self.children[infoset.parent_sequence_id].append(infoset)

    def all_deterministic_strategies(self, root_seq=None):
        if root_seq is None:
            root_seq = self.root_sequence_id

        ans = []
        if len(self.children[root_seq]) > 0:  # The root seq is NOT terminal:
            ans = [
                sum(pieces)
                for pieces in itertools.product(
                    *[self.all_deterministic_strategies_infoset(child)
                        for child in self.children[root_seq]])]
            for v in ans:
                v[root_seq] = 1.0
        else:
            v = np.zeros(self.n_sequences)
            v[root_seq] = 1.0
            ans += [v]
        return ans

    def all_deterministic_strategies_infoset(self, infoset):
        ans = []
        for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
            ans += self.all_deterministic_strategies(seq)
        return ans
    
    def is_sf_strategy(self, x):
        """Checks that x is a valid sequence-form strategy"""
        if not (x >= 0.0).all():
            return False
        if abs(x[self.root_sequence_id] - 1.0) > 1e-9:
            return False
        for infoset in self.infosets:
            s = -x[infoset.parent_sequence_id]
            for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                s += x[seq]
            if abs(s) > 1e-9:
                return False
        return True

    def best_response_value(self, gradient):
        g = gradient.copy()  # We do not want to modify the gradient that was passed
        for infoset in self.infosets:
            slice = g[infoset.start_sequence_id : (infoset.end_sequence_id + 1)]
            g[infoset.parent_sequence_id] += slice.max()
        return g[self.root_sequence_id]

class Game(object):
    class PayoffEntry(object):
        def __init__(self, capnp_obj):
            self.payoffs = list(capnp_obj.payoffs)
            self.chanceFactor = capnp_obj.chanceFactor
            self.sequences = list(capnp_obj.sequences)
            # print(self.payoffs, capnp_obj.sequences)
        
    def __init__(self, path):
        with open(path) as capnp_file:
            capnp_obj = game_capnp.Game.read(capnp_file)
            # print(capnp_obj.treeplexes)
            self.tpxs = [Treeplex(capnp_treeplex) for capnp_treeplex in capnp_obj.treeplexes]

            self.payoff_matrix_entries = [
                self.PayoffEntry(obj) for obj in capnp_obj.payoffMatrix.entries]
            
        self.n_players = len(self.tpxs)
        self.payoff_matrix_nnz = len(self.payoff_matrix_entries)
        self.notes = capnp_obj.notes

    def utility_gradient(self, player, strategies):
        """Returns the gradient of the utility of the given player, given the dictionary of strategies for all other players.
        
        As usual, players are 0-based.
        """
        grad = np.zeros(self.tpxs[player].n_sequences)
        #print(len(grad))
        # print(len(self.payoff_matrix_entries))
        # print(strategies)
        for entry in self.payoff_matrix_entries:
            # if player == 1:
            #     print("Looking at entry: ", entry, "  payoff: ", entry.payoffs)
            #     print("Entry.sequences (indexed by player p): ", entry.sequences)
            reach = entry.chanceFactor
            # if player == 1:
            #     print("CHANCE: ", reach)
            for p in range(self.n_players):
                if p != player:
                    # if player == 1:
                    #     # print("Multiplying by ", strategies[p][entry.sequences[p]], " for seq: ", entry.sequences[p])
                    #     # print("    Other player seq: ", entry.sequences[1-p])
                    #     print("Getting the strategy from seq of other player at seq: ", entry.sequences[p])
                    #     print("   This is updating the player 1 seq: ", entry.sequences[player])
                    reach *= strategies[p][entry.sequences[p]]
            grad[entry.sequences[player]] += entry.payoffs[player] * reach
        
        return grad
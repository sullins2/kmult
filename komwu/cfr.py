import numpy as np

class Cfr(object):
    TOL = 1e-6  # If the sum of the local regret is below TOL, the sum is treated like zero
                # and no normalization happens.

    def __init__(self, tpx, plus=False):
        self.tpx = tpx
        self.plus = plus
        self.buffer = np.zeros(tpx.n_sequences)

        # Used to track regret
        self.sum_gradients = np.zeros(tpx.n_sequences)
        self.sum_ev = 0.0

    def next_strategy(self):
        ans = np.maximum(self.buffer, 0.0)

        # Iterating in reverse over the infosets corresponds to a top-down traversal
        ans[self.tpx.root_sequence_id] = 1.0
        for infoset in self.tpx.infosets[::-1]:
            slice = ans[infoset.start_sequence_id : (infoset.end_sequence_id + 1)]
            if slice.sum() > self.TOL:
                slice /= slice.sum()
                slice *= ans[infoset.parent_sequence_id]
            else:
                slice.fill(ans[infoset.parent_sequence_id] / infoset.n_sequences)
        assert self.tpx.is_sf_strategy(ans)
        return ans

    def observe_gradient(self, gradient):
        self.sum_gradients += gradient
        self.sum_ev += gradient.dot(self.next_strategy())

        # The information sets are sorted in a bottom-up way
        for infoset in self.tpx.infosets:
            local_strat = np.maximum(self.buffer[infoset.start_sequence_id : (infoset.end_sequence_id + 1)], 0.0)
            grad_slice = gradient[infoset.start_sequence_id : (infoset.end_sequence_id + 1)]

            if local_strat.sum() > self.TOL:
                ev = local_strat.dot(grad_slice) / local_strat.sum()
            else:
                ev = grad_slice.sum() / infoset.n_sequences
            gradient[infoset.parent_sequence_id] += ev

            self.buffer[infoset.start_sequence_id : (infoset.end_sequence_id + 1)] += grad_slice
            self.buffer[infoset.start_sequence_id : (infoset.end_sequence_id + 1)] -= ev

        if self.plus:
            self.buffer = np.maximum(self.buffer, 0.0)

    def regret(self):
        return self.tpx.best_response_value(self.sum_gradients) - self.sum_ev
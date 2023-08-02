import numpy as np
from scipy.special import logsumexp

class m2wu(object):
    def __init__(self, tpx, eta, mu, N):
        self.tpx = tpx
        self.eta = eta      # Learning rate
        self.mu = mu
        self.N = N
        self.last_gradient = np.zeros(tpx.n_sequences)
        # To improve numerical stability, we store b in logarithmic form.
        self.b = np.zeros(tpx.n_sequences)
        self.b_xi = np.zeros(tpx.n_sequences)
        self._compute_x(t=0)
        self.x_ref = self.x.copy()

        # Used to track regret
        self.sum_gradients = np.zeros(tpx.n_sequences)
        self.sum_ev = 0.0

    def next_strategy(self):
        return self.x

    def next_strategy_ref(self):
        return self.x_ref

    def update_strategy_ref(self, t):
        if t % self.N == 0:
            self.x_ref = self.x.copy()

    def observe_gradient(self, gradient, t):

        self.sum_gradients += gradient
        self.sum_ev += gradient.dot(self.next_strategy())

        # We use as prediction for the next gradient the current gradient. So, the
        # optimistic gradient is 2 * current gradient - previous gradient.
        # optimistic_gradient = 2 * gradient - 1*self.last_gradient
        # self.last_gradient = gradient
        # values = np.exp(self.eta * (utility + self.mu * (self.ref_strategy - self.strategy) / self.strategy)) * self.strategy
        value = (self.eta * (gradient + self.mu * (self.next_strategy_ref() - self.next_strategy()) / self.next_strategy()))
        # self.last_gradient = gradient
        self.b += value


        self._compute_x(t)

    def _compute_x(self,t):
        """Compute the strategy x^t according to (10)"""

        # Step 1. We compute the values K_j(b,1) for all infosets j in cumulative
        # O(|\Sigma|) time, using (14). We store K_j in log form.
        #
        # The order of the K_j values follows the order of the infosets in tpx.infosets.
        # NB: since b is stored in log form, we need to use logsumexp to sum the values.
        # print("CALLEDD")
        K_j = [None] * self.tpx.n_infosets
        # print("Printing infosets in order:")
        # for infoset_id, infoset in enumerate(self.tpx.infosets):
            # print("ID: ", infoset_id, " INFOSET: ", infoset.start_sequence_id, infoset.end_sequence_id)
        #
        #     K_j[infoset_id] = logsumexp([
        #         self.b[seq] + sum([K_j[child_infoset.infoset_id] for child_infoset in self.tpx.children[seq]])
        #         for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1)
        #     ])
        #     # print(self.b)
        #     print(infoset.start_sequence_id)
        #     print(K_j[infoset_id])

        # De pythonic version of above
        for infoset_id in range(len(self.tpx.infosets)):
            infoset = self.tpx.infosets[infoset_id]
            seq_values = []
            for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                child_values = []
                for child_infoset in self.tpx.children[seq]:
                    child_values.append(K_j[child_infoset.infoset_id])
                seq_value = self.b[seq] + sum(child_values)
                seq_values.append(seq_value)
            K_j[infoset_id] = logsumexp(seq_values)
        #     print(" Set K_j to: ", K_j[infoset_id], " infosetID: ", infoset_id)
        # print("")

        # Step 2. We compute the ratio K(b, e_root) / K(b, 1).
        # We store the values in log form.
        #
        # We start from K(b, 1) using Theorem 5.2
        K_b_1 = self.b[self.tpx.root_sequence_id] + \
             sum([K_j[infoset.infoset_id] for infoset in self.tpx.children[self.tpx.root_sequence_id]])
        # K(b, e_root) = 0 by definition.

        # Step 3. We compute
        #            y_ja := 1 - K(b, ebar_ja) / K(b, 1)
        # for all sequence ja in top-down order in accordance with Proposition 5.3.
        #
        # We store y_ja in log form. So in particular y_root = log(1 - 0) = 0.
        y = np.zeros(self.tpx.n_sequences)
        for infoset in self.tpx.infosets[::-1]:
            for sequence_id in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                # Proposition 5.3 in logarithmic form

                y[sequence_id] = y[infoset.parent_sequence_id] \
                        + self.b[sequence_id] + sum([K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
                        - K_j[infoset.infoset_id]


        self.x = np.exp(y)
        self.update_strategy_ref(t)
        #assert self.tpx.is_sf_strategy(self.x)

    def regret(self):
        return self.tpx.best_response_value(self.sum_gradients) - self.sum_ev
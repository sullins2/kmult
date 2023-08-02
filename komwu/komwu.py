import numpy as np
from scipy.special import logsumexp

class Komwu(object):
    def __init__(self, tpx, eta, player, B_STUFF):
        self.tpx = tpx
        self.eta = eta      # Learning rate


        self.B_STUFF = B_STUFF
        self.last_gradient = np.zeros(tpx.n_sequences)

        # To improve numerical stability, we store b in logarithmic form.
        self.b = np.zeros(tpx.n_sequences)
        self.b_xi = np.zeros(tpx.n_sequences)
        self._compute_x(t=0, first_step=False)

        self.opt_level = np.ones(tpx.n_sequences) * 2.0

        self.b_store = np.zeros(tpx.n_sequences)
        self.b_count = 0
        self.b_max = 20
        self.b_half = 10

        # Used to track regret
        self.sum_gradients = np.zeros(tpx.n_sequences)
        self.sum_ev = 0.0

    def next_strategy(self):
        # return [0.33333333 0.33333333 0.66666667 0.33333333 0.33333333 0.33333333 0.66666667 0.33333333 0.33333333 0.33333333 0.66666667 0.33333333 1. ]
        # return [0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 1.0]
        return self.x

    def next_strategy_xi(self):
        return self.x_xi

    def observe_gradient(self, gradient, t, first_step, komwu):

        if first_step == False:
            self.sum_gradients += gradient
            self.sum_ev += gradient.dot(self.next_strategy())

        # We use as prediction for the next gradient the current gradient. So, the 
        # optimistic gradient is 2 * current gradient - previous gradient.
        # optimistic_gradient = 2 * gradient - 1*self.last_gradient
        # self.last_gradient = gradient
        if first_step:
            xi = 100.0
            self.b_xi = self.b.copy()
            self.b_xi += xi * gradient
            # Try optimism here?
            # optimistic_gradient = 2 * gradient - 1 * self.last_gradient
            # self.last_gradient = gradient
            # self.b_xi += xi * optimistic_gradient

        else:
            eta = 1.0
            if komwu == False:
                self.b += eta * gradient
                # optimistic_gradient = 2 * gradient - 1 * self.last_gradient
                # self.last_gradient = gradient
                # self.b += eta * optimistic_gradient
            else:
                # # print("--------------Here-------------------")
                # # print(gradient)
                # # print(self.last_gradient)
                # # print(gradient - self.last_gradient)
                # dif = 0.001
                # temp = np.abs(gradient - self.last_gradient)
                #
                # last = self.opt_level.copy() - 1
                #
                # for ii in range(len(temp)):
                #     if temp[ii] <= dif:
                #         self.opt_level[ii] += 1.0
                #         if self.opt_level[ii] > 20.0:
                #             self.opt_level[ii] = 20.0
                #     else:
                #         self.opt_level[ii] -= 2.0
                #         if self.opt_level[ii] < 2.0:
                #             self.opt_level[ii] = 2.0
                # # print(self.opt_level)
                # optimistic_gradient = self.opt_level * gradient - last * self.last_gradient
                # self.last_gradient = gradient
                # self.b += eta * optimistic_gradient

                # print("-------------------------------------")

                # ONLY THIS IN REGULAR VERSION
                optimistic_gradient = 2 * gradient - 1 * self.last_gradient
                self.last_gradient = gradient
                self.b += eta * optimistic_gradient
                if self.b_count >= self.b_half:
                    self.b_store += eta * optimistic_gradient

                if self.b_count == self.b_max and self.B_STUFF:
                    self.b = self.b_store.copy()
                    # self.b_store *= 0
                    self.b_count = 0

                self.b_count += 1

        self._compute_x(t, first_step)

    def _compute_x(self,t, first_step):
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

        if t > 1:
            mod = t // 100
            ent = -5.0 / np.log(t + 2.0)
            for infoset_id in range(len(self.tpx.infosets)):
                infoset = self.tpx.infosets[infoset_id]
                pols = []
                sum_vals = 0.0
                for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                    pols.append(self.x[seq])
                pols_sum = sum(pols)
                if pols_sum <= 0:
                    pols_sum = 1e-32
                final_pols = np.array(pols) / pols_sum
                for ii in range(len(final_pols)):
                    pol = final_pols[ii]
                    if pol <= 0:
                        pol = 1e-32
                    sum_vals += pol * np.log(pol)
                for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                    self.b[seq] += ent*(0 - sum_vals)

        # De pythonic version of above
        for infoset_id in range(len(self.tpx.infosets)):
            infoset = self.tpx.infosets[infoset_id]
            seq_values = []
            for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                child_values = []
                for child_infoset in self.tpx.children[seq]:
                    child_values.append(K_j[child_infoset.infoset_id])
                if first_step:
                    seq_value = self.b_xi[seq] + sum(child_values)
                else:
                    seq_value = self.b[seq] + sum(child_values)
                seq_values.append(seq_value)
            K_j[infoset_id] = logsumexp(seq_values)

        #     print(" Set K_j to: ", K_j[infoset_id], " infosetID: ", infoset_id)
        # print("")


        # what needs to be cached if not updating children at infoset I
        # other person is changing what they are doing, what do I need to track


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
                if first_step:
                    y[sequence_id] = y[infoset.parent_sequence_id] \
                        + self.b_xi[sequence_id] + sum([K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
                                     - K_j[infoset.infoset_id]
                else:
                    y[sequence_id] = y[infoset.parent_sequence_id] \
                        + self.b[sequence_id] + sum([K_j[child.infoset_id] for child in self.tpx.children[sequence_id]]) \
                        - K_j[infoset.infoset_id]


        if first_step:
            self.x_xi = np.exp(y)
        else:
            self.x = np.exp(y)
        #assert self.tpx.is_sf_strategy(self.x)

    def regret(self):
        return self.tpx.best_response_value(self.sum_gradients) - self.sum_ev
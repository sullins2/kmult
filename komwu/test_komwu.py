import unittest

from game import Game
from komwu import Komwu
import numpy as np
from scipy.special import softmax

# This test compares running KOMWU and running OMWU on the vertices of the sequence-form
# polytope using random gradients.

class VertexOmwu(object):
    def __init__(self, tpx, eta):
        self.tpx = tpx
        self.eta = eta
        # 13 (4 deterministic for J, 4 for Q, 4 for K, one at root)
        self.vertices = self.tpx.all_deterministic_strategies()
        print(self.vertices)
        # 27
        self.n_vertices = len(self.vertices)
        print(self.n_vertices)
        self.last_gradient = np.zeros(self.tpx.n_sequences)

        self.V = np.array(self.vertices).T
        
        # Distribution over vertices, stored in unnormalized log form
        self.lam = np.zeros(self.n_vertices)
        self._compute_x()
    
    def next_strategy(self):
        return self.x

    def observe_gradient(self, gradient):
        optimistic_gradient = 2 * gradient - self.last_gradient
        self.last_gradient = gradient

        self.lam += self.eta * (self.V.T @ optimistic_gradient)
        self._compute_x()
    
    def _compute_x(self):
        self.x = self.V @ softmax(self.lam)

class TestKowmu(unittest.TestCase):
    def setUp(self):
        self.game = Game("game_instances/K23.game")
        self.eta = 0.12345

    def testRandomGradients(self): 
        tpx = self.game.tpxs[0]
        komwu = Komwu(tpx, eta=self.eta, player=1)
        vomwu = VertexOmwu(tpx, eta=self.eta)

        for t in range(1, 101):
            self.assertTrue(np.allclose(
                komwu.next_strategy(),
                vomwu.next_strategy()
            ))

            gradient = 2.0 * np.random.rand(tpx.n_sequences) - 1.0
            komwu.observe_gradient(gradient, t=0)
            vomwu.observe_gradient(gradient)


if __name__ == '__main__':
    unittest.main()
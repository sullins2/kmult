import numpy as np
import matplotlib.pyplot as plt


class algs(object):

    def __init__(self, T, eta):
        self.omwu_list = []
        self.omwu_list1 = []
        self.omwu_list2 = []
        self.omwu_list3 = []
        self.flbr_list = []
        self.flbr_opt_list = []
        self.iterations = [t for t in range(T)]
        self.T = T
        self.eta = eta
        self.p10 = 0
        self.payoffs1 = None
        self.payoffs2 = None

        self.last_p1 = None
        self.last_p2 = None



        game = np.array([[2 / 3.0, -2 / 3.0], [-3 / 3.0, 3 / 3.0]])
        self.game = np.array([[1.0 / 5.0, 2 / 5.0, 4 / 5.0], [2 / 5.0, 1 / 5.0, 2 / 5.0], [1 / 5.0, 2 / 5.0, 5 / 5.0]])
        self.game = np.array([[1 / 6.0, -2 / 6.0, 4 / 6.0, 2 / 6.0, 4 / 6.0, 4 / 6.0],
                         [-1 / 6.0, 3 / 6.0, 2 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0],
                         [2 / 6.0, 4 / 6.0, -1 / 6.0, 1 / 6.0, 3 / 6.0, 2 / 6.0],
                         [3 / 6.0, -2 / 6.0, 1 / 6.0, -3 / 6.0, -2 / 6.0, 3 / 6.0],
                         [3 / 6.0, 4 / 6.0, -2 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
                         [1 / 6.0, -2 / 6.0, -2 / 6.0, 2 / 6.0, 1 / 6.0, 2 / 6.0]])

        # self.game = np.array(
        #     [[1.0 / 5.0, -2 / 5.0, 2 / 5.0],
        #      [-2 / 5.0, 1 / 5.0, 2 / 5.0],
        #      [1 / 5.0, 2 / 5.0, -5 / 5.0]])

        #-2 1 0 3 -1 2 -3
        # 3 0 1 -2 2 -1 0
        # 0 3 -2 1 1 -3 2
        # -3 -1 2 1 -1 0 3
        # 2 2 -1 -3 0 1 1
        # -1 -3 0 2 3 1 1
        # -1 -1 -1 -1 -1 -1 -1
        # self.game = np.array([[-2 / 3.0, 1 / 3.0, 0 / 3.0, 3 / 3.0, -1 / 3.0, 2 / 3.0, -3 / 3.0],
        #                  [3 / 3.0, 0 / 3.0, 1 / 3.0, -2 / 3.0, 2 / 3.0, -1 / 3.0, 0 / 3.0],
        #                  [0 / 3.0, 3 / 3.0, -2 / 3.0, 1 / 3.0, 1 / 3.0, -3 / 3.0, 2 / 3.0],
        #                  [-3 / 3.0, -1 / 3.0, 2 / 3.0, 1 / 3.0, -1 / 3.0, 0 / 3.0, 3 / 3.0],
        #                  [2 / 3.0, 2 / 3.0, -1 / 3.0, -3 / 3.0, 0 / 3.0, 1 / 3.0, 1 / 3.0],
        #                  [-1 / 3.0, -3 / 3.0, 0 / 3.0, 2 / 3.0, 3 / 3.0, 1 / 3.0, 1 / 3.0],
        #                  [-1 / 3.0, -1 / 3.0, -1 / 3.0, -1 / 3.0, -1 / 3.0, -1 / 3.0, -1 / 3.0]])

        # self.game = np.array([[0, -0.1, 0.2],
        #                   [0.1, 0, -0.1],
        #                   [-0.2, 0.1, 0]])

        self.game = np.array([[-2/10.0,1/10.0,0/10.0,1/10.0,-1/10.0,1/10.0,-10/10.0,2/10.0,2/10.0,2/10.0],
                [10/10.0,0/10.0,1/10.0,-1/10.0,1/10.0,-1/10.0,0/10.0,1/10.0,-2/10.0,1.0/10.0],
                [0/10.0,1/10.0,-2/10.0,-1/10.0,4/10.0,-10/10.0,2/10.0,0/10.0,1/10.0,4/10.0],
                [-10/10.0,-1/10.0,2/10.0,7/10.0,-2/10.0,0/10.0,10/10.0,2/10.0,-10/10.0,0.0],
                [2/10.0,2/10.0,-1/10.0,-10/10.0,5/10.0,1/10.0,1/10.0,-2/10.0,-2/10.0,0.0],
                [-1/10.0,10/10.0,0/10.0,2/10.0,10/10.0,1/10.0,7/10.0,7/10.0,2/10.0,0.0],
                [-1/10.0,-4/10.0,-1/10.0,-1/10.0,-1/10.0,-1/10.0,-1/10.0,10/10.0,-2/10.0,5/10.0],
                [2/10.0,-1/10.0,1/10.0,-2/10.0,-1/10.0,-2/10.0,1/10.0,2/10.0,1/10.0,2/10.0],
                [1/10.0,-2/10.0,1/10.0,-2/10.0,1/10.0,-1/10.0,1/10.0,2/10.0,-2/10.0,0.0],
                [1/10.0,4/10.0,1/10.0,1/10.0,-2/10.0,5/10.0,-1/10.0,1/10.0,-1/10.0,1.0 / 10.0]])


        def create_normalized_matrix(n, num_negative_entries=0):
            # Step 1: Create a random matrix with values between -1 and 1
            matrix = (np.random.rand(self.num_actions0, self.num_actions1) - 0.0) #* 0.1

            # Step 2: Normalize each row to ensure payoffs sum to one
            # row_sums = np.abs(matrix).sum(axis=1, keepdims=True)
            # normalized_matrix = matrix / row_sums

            # Step 3: Set a random number of entries to negative
            # if num_negative_entries > 0:
            #     indices = np.random.choice(n * n, num_negative_entries, replace=False)
            #     normalized_matrix.flat[indices] *= -1

            return matrix
        np.random.seed(12171) #111, 1111, 112
        n = -1
        self.num_actions0 = 10
        self.num_actions1 = 11
        matrix_game = create_normalized_matrix(n)
        print("MATRIX GAME")
        print(matrix_game)
        self.game = matrix_game

        self.record = 8
        self.num_actions = len(self.game[0])
        self.b0 = np.zeros(self.num_actions0)
        self.b1 = np.zeros(self.num_actions1)
        self.b0_store = np.zeros(self.num_actions0)
        self.b1_store = np.zeros(self.num_actions1)

    def plot_graph(self):
        if len(self.flbr_list) > 0:
            plt.plot(self.iterations, self.flbr_list, label='FLBR', color='blue')
        if len(self.omwu_list1) > 0:
            plt.plot(self.iterations, self.omwu_list1, label='OMWU', color='red')
        if len(self.omwu_list2) > 0:
            plt.plot(self.iterations, self.omwu_list2, label='OMWU_b', color='green')
        if len(self.omwu_list3) > 0:
            plt.plot(self.iterations, self.omwu_list3, label='FLBR_b', color='orange')
        if len(self.flbr_opt_list) > 0:
            plt.plot(self.iterations, self.flbr_opt_list, label='FLBR_OPT', color='green')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Policy')
        # plt.title('Optimization Progress')
        plt.legend()
        plt.show()

    def run_omwu(self):
        self.b0 = np.zeros(self.num_actions0)
        self.b1 = np.zeros(self.num_actions1)
        self.b0_store = np.zeros(self.num_actions0)
        self.b1_store = np.zeros(self.num_actions1)

        p1 = np.array([1.0 / self.num_actions0 for _ in range(self.num_actions0)])
        p2 = np.array([1.0 / self.num_actions1 for _ in range(self.num_actions1)])

        last_payoffs1 = 0.0
        last_payoffs2 = 0.0
        b_count = 1

        for t in range(self.T):
            # if t % 500 == 0:
            #     print("Iteration:", t)
            # Calculate the payoffs for each player
            payoffs1 = np.dot(self.game, p2)
            payoffs2 = -np.dot(self.game.T, p1)
            if t == self.T-1: # TO SEE FINAL GAME VALUE
                print("PAYOFFS1:", sum(payoffs1 * p1))
                print("PAYOFFS2:", sum(payoffs2 * p2))

            self.payoffs1 = payoffs1
            self.payoffs2 = payoffs2

            # mod = self.T // 10000000
            # ent = 0 #-0.1 / (mod + 1)

            # sum_vals = 0.0
            # for i in range(len(payoffs1)):
            #     pol = p1[i]
            #     if pol <= 0:
            #         pol = 1e-32
            #     sum_vals += pol * np.log(pol)
            # for i in range(len(payoffs1)):
            #     self.b0[i] += ent * -sum_vals

            # sum_vals = 0.0
            # for i in range(len(payoffs2)):
            #     pol = p1[i]
            #     if pol <= 0:
            #         pol = 1e-32
            #     sum_vals += pol * np.log(pol)
            # for i in range(len(payoffs2)):
            #     self.b1[i] += ent * -sum_vals
                # if t > self.T - 10:
                # # if t > 0 and t < 10:
                #     print(ent * -sum_vals)


            # KL = 0 #-5.0
            # if self.last_p1 is not None:
            #     sum_vals = 0.0
            #     for i in range(len(payoffs1)):
            #         pol = p1[i]
            #         if pol <= 0:
            #             pol = 1e-32
            #         if self.last_p1[i] <= 1e-32:
            #             self.last_p1[i] = 1e-32
            #         sum_vals += -pol * np.log(pol / self.last_p1[i])
            #     for i in range(len(payoffs1)):
            #         self.b0[i] += KL * sum_vals
            # self.last_p1 = p1.copy()
            #
            #
            # if self.last_p2 is not None:
            #     sum_vals = 0.0
            #     for i in range(len(payoffs2)):
            #         pol = p2[i]
            #         if pol <= 0:
            #             pol = 1e-32
            #         if self.last_p2[i] <= 1e-32:
            #             self.last_p2[i] = 1e-32
            #         sum_vals += -pol * np.log(pol / self.last_p2[i])
            #     for i in range(len(payoffs2)):
            #         self.b1[i] += KL * sum_vals
            # self.last_p2 = p2.copy()

            ETA = self.eta
            self.b0 += ETA * (2.0 * payoffs1 - 1.0*last_payoffs1)
            self.b1 += ETA * (2.0 * payoffs2 - 1.0*last_payoffs2)
            b_count += 1
            last_payoffs1 = payoffs1.copy()
            last_payoffs2 = payoffs2.copy()

            max_b0 = np.max(self.b0)
            b0 = self.b0 - max_b0
            p1 = np.exp(1.0 * b0)
            max_b1 = np.max(self.b1)
            b1 = self.b1 - max_b1
            p2 = np.exp(1.0 * b1)

            # Normalize the strategy profile
            p1 = p1 / np.sum(p1)
            p2 = p2 / np.sum(p2)
            self.omwu_list1.append(p1[self.record])# / (t + 1))

            self.p10 = p1
            self.p1 = p1
            self.p2 = p2

########################################################################################

    def run_omwu_b(self):
        self.b0 = np.zeros(self.num_actions0)
        self.b1 = np.zeros(self.num_actions1)
        self.b0_store = np.zeros(self.num_actions0)
        self.b1_store = np.zeros(self.num_actions1)

        p1 = np.array([1.0 / self.num_actions0 for _ in range(self.num_actions0)])
        p2 = np.array([1.0 / self.num_actions1 for _ in range(self.num_actions1)])
        first_it = None
        last_payoffs1 = 0.0
        last_payoffs2 = 0.0
        b_count = 1
        for t in range(self.T):
            # Calculate the payoffs for each player
            payoffs1 = np.dot(self.game, p2)
            payoffs2 = -np.dot(self.game.T, p1)
            if t == self.T-1:
                print("PAYOFFS1:", sum(payoffs1 * p1))
                print("PAYOFFS2:", sum(payoffs2 * p2))

            self.payoffs1 = payoffs1
            self.payoffs2 = payoffs2

            # mod = self.T // 10000000
            # ent = 0 #-0.1 / (mod + 1)
            #
            # sum_vals = 0.0
            # for i in range(len(payoffs1)):
            #     pol = p1[i]
            #     if pol <= 0:
            #         pol = 1e-32
            #     sum_vals += pol * np.log(pol)
            # for i in range(len(payoffs1)):
            #     self.b0[i] += ent * -sum_vals
            #
            # sum_vals = 0.0
            # for i in range(len(payoffs2)):
            #     pol = p1[i]
            #     if pol <= 0:
            #         pol = 1e-32
            #     sum_vals += pol * np.log(pol)
            # for i in range(len(payoffs2)):
            #     self.b1[i] += ent * -sum_vals
                # if t > self.T - 10:
                # # if t > 0 and t < 10:
                #     print(ent * -sum_vals)


            # KL = 0 #-5.0
            # if self.last_p1 is not None:
            #     sum_vals = 0.0
            #     for i in range(len(payoffs1)):
            #         pol = p1[i]
            #         if pol <= 0:
            #             pol = 1e-32
            #         if self.last_p1[i] <= 1e-32:
            #             self.last_p1[i] = 1e-32
            #         sum_vals += -pol * np.log(pol / self.last_p1[i])
            #     for i in range(len(payoffs1)):
            #         self.b0[i] += KL * sum_vals
            # self.last_p1 = p1.copy()


            # if self.last_p2 is not None:
            #     sum_vals = 0.0
            #     for i in range(len(payoffs2)):
            #         pol = p2[i]
            #         if pol <= 0:
            #             pol = 1e-32
            #         if self.last_p2[i] <= 1e-32:
            #             self.last_p2[i] = 1e-32
            #         sum_vals += -pol * np.log(pol / self.last_p2[i])
            #     for i in range(len(payoffs2)):
            #         self.b1[i] += KL * sum_vals
            # self.last_p2 = p2.copy()

            ETA = self.eta
            self.b0 += ETA * (2.0 * payoffs1 - 1.0*last_payoffs1)
            self.b1 += ETA * (2.0 * payoffs2 - 1.0*last_payoffs2)
            b_count += 1
            if b_count >= 10:
                self.b0_store += ETA * (2.0 * payoffs1 - 1.0 * last_payoffs1)
                self.b1_store += ETA * (2.0 * payoffs2 - 1.0 * last_payoffs2)

            last_payoffs1 = payoffs1.copy()
            last_payoffs2 = payoffs2.copy()

            max_b0 = np.max(self.b0)
            b0 = self.b0 - max_b0
            p1 = np.exp(1.0 * b0)
            max_b1 = np.max(self.b1)
            b1 = self.b1 - max_b1
            p2 = np.exp(1.0 * b1)

            if b_count == 20:
                self.b0 = self.b0_store.copy()
                self.b1 = self.b1_store.copy()
                b_count = 1

            p1 = p1 / np.sum(p1)
            p2 = p2 / np.sum(p2)
            self.omwu_list2.append(p1[self.record])# / (t + 1))

            self.p10 = p1
            self.p1 = p1
            self.p2 = p2


########################################################################################

    # def run_flbr(self):
    #
    #     self.b0 = np.zeros(self.num_actions)
    #     self.b1 = np.zeros(self.num_actions)
    #     self.b0_store = np.zeros(self.num_actions)
    #     self.b1_store = np.zeros(self.num_actions)
    #     # Initialize the strategy profile
    #     p1 = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     p2 = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     # p1[0] -= 0.11
    #     # p1[2] += 0.11
    #
    #     p1_hat = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     p2_hat = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     last_payoffs1 = 0.0
    #     last_payoffs2 = 0.0
    #     first_it = None
    #     xi = 15.0
    #     b_count = 1
    #     # Run the Multiplicative Weights Update algorithm for T iterations
    #     for t in range(self.T):
    #         # Calculate the payoffs for each player
    #         payoffs1 = np.dot(self.game, p2)
    #         payoffs2 = -np.dot(self.game.T, p1)
    #         if t == self.T-1:
    #             print("PAYOFFS1:", sum(payoffs1 * p1))
    #             print("PAYOFFS2:", sum(payoffs2 * p2))
    #
    #         self.b0_temp = self.b0.copy()
    #         self.b1_temp = self.b1.copy()
    #         self.b0_temp += xi * payoffs1
    #         self.b1_temp += xi * payoffs2
    #
    #         p1_hat = np.exp(self.b0_temp)
    #         p2_hat = np.exp(self.b1_temp)
    #
    #         # Calculate payoffs with pi_hats
    #         payoffs1_hat = np.dot(self.game, p2_hat)
    #         payoffs2_hat = -np.dot(self.game.T, p1_hat)
    #
    #         self.b0 += self.eta * payoffs1_hat
    #         self.b1 += self.eta * payoffs2_hat
    #
    #         # Calculate pi^t+1
    #         p1 = np.exp(self.b0)
    #         p2 = np.exp(self.b1)
    #
    #         b_count += 1
    #         if b_count >= 15:
    #             self.b0_store += self.eta * payoffs1_hat
    #             self.b1_store += self.eta * payoffs2_hat
    #
    #         # last_payoffs1 = payoffs1.copy()
    #         # last_payoffs2 = payoffs2.copy()
    #
    #         if b_count == -30:
    #             self.b0 = self.b0_store.copy()
    #             self.b1 = self.b1_store.copy()
    #             b_count = 1
    #
    #
    #         # Normalize the strategy profile
    #         p1 = p1 / np.sum(p1)
    #         p2 = p2 / np.sum(p2)
    #         self.flbr_list.append(p1[self.record])

        # Print the final strategy profile and expected payoffs for each player
        # print("Final strategy profile:")
        # print("Player 1:", p1)
        # print("Player 2:", p2)
        # print("Expected payoffs:")
        # print("Player 1:", np.dot(payoffs1, p1))
        # print("Player 2:", np.dot(payoffs2, p2))

    def run_flbr(self):

        self.b0 = np.zeros(self.num_actions0)
        self.b1 = np.zeros(self.num_actions1)
        self.b0_store = np.zeros(self.num_actions0)
        self.b1_store = np.zeros(self.num_actions1)

        # Initialize the strategy profile
        p1 = np.array([1.0 / self.num_actions for _ in range(self.num_actions0)])
        p2 = np.array([1.0 / self.num_actions for _ in range(self.num_actions1)])
        # p1[0] -= 0.11
        # p1[2] += 0.11

        p1_hat = np.array([1.0 / self.num_actions0 for _ in range(self.num_actions0)])
        p2_hat = np.array([1.0 / self.num_actions1 for _ in range(self.num_actions1)])

        payoffs1_prev = None
        payoffs2_prev = None
        first_it = None
        xi = 50.0
        # Run the Multiplicative Weights Update algorithm for T iterations
        for t in range(self.T):
            # Calculate the payoffs for each player
            payoffs1 = np.dot(self.game, p2)
            payoffs2 = -np.dot(self.game.T, p1)
            if t == self.T-1:
                print("PAYOFFS1:", sum(payoffs1 * p1))
                print("PAYOFFS2:", sum(payoffs2 * p2))

            # Calculate pi_hats
            p1_hat = p1 * np.exp(xi * payoffs1)
            p2_hat = p2 * np.exp(xi * payoffs2)

            # Calculate payoffs with pi_hats
            payoffs1_hat = np.dot(self.game, p2_hat)
            payoffs2_hat = -np.dot(self.game.T, p1_hat)

            # Calculate pi^t+1
            payoffs1 = payoffs1 - np.max(payoffs1_hat)
            payoffs2 = payoffs1 - np.max(payoffs2_hat)

            p1 = p1 * np.exp(self.eta * payoffs1_hat)
            p2 = p2 * np.exp(self.eta * payoffs2_hat)

            # Normalize the strategy profile
            p1 = p1 / np.sum(p1 + 1e-10)
            p2 = p2 / np.sum(p2 + 1e-10)

            self.flbr_list.append(p1[self.record])

        # Print the final strategy profile and expected payoffs for each player
        # print("Final strategy profile:")
        # print("Player 1:", p1)
        # print("Player 2:", p2)
        # print("Expected payoffs:")
        # print("Player 1:", np.dot(payoffs1, p1))
        # print("Player 2:", np.dot(payoffs2, p2))

    def run_flbr_b(self):

        self.b0 = np.zeros(self.num_actions0)
        self.b1 = np.zeros(self.num_actions1)
        self.b0_store = np.zeros(self.num_actions0)
        self.b1_store = np.zeros(self.num_actions1)

        # Initialize the strategy profile
        p1 = np.array([1.0 / self.num_actions0 for _ in range(self.num_actions0)])
        p2 = np.array([1.0 / self.num_actions1 for _ in range(self.num_actions1)])
        # p1[0] -= 0.11
        # p1[2] += 0.11

        p1_hat = np.array([1.0 / self.num_actions0 for _ in range(self.num_actions0)])
        p2_hat = np.array([1.0 / self.num_actions1 for _ in range(self.num_actions1)])

        payoffs1_prev = None
        payoffs2_prev = None
        first_it = None
        xi = 5.0
        b_count = 0
        # Run the Multiplicative Weights Update algorithm for T iterations
        for t in range(self.T):
            # Calculate the payoffs for each player

            b_count += 1
            payoffs1 = np.dot(self.game, p2)
            payoffs2 = -np.dot(self.game.T, p1)
            if t == self.T-1:
                print("PAYOFFS1:", sum(payoffs1 * p1))
                print("PAYOFFS2:", sum(payoffs2 * p2))

            # Calculate pi_hats
            p1_hat = p1 * np.exp(xi * payoffs1)
            p2_hat = p2 * np.exp(xi * payoffs2)

            # Calculate payoffs with pi_hats
            payoffs1_hat = np.dot(self.game, p2_hat)
            payoffs2_hat = -np.dot(self.game.T, p1_hat)


            # Calculate pi^t+1
            p1 = p1 * np.exp(self.eta * payoffs1_hat)
            p2 = p2 * np.exp(self.eta * payoffs2_hat)
            if b_count == 1:
                p1_b = p1.copy() #* np.exp(self.eta * payoffs1_hat)
                p2_b = p2.copy() #* np.exp(self.eta * payoffs2_hat)
            if b_count > 10:
                p1_b = p1_b * np.exp(self.eta * payoffs1_hat)
                p2_b = p2_b * np.exp(self.eta * payoffs2_hat)


            # Normalize the strategy profile
            p1 = p1 / np.sum(p1)
            p2 = p2 / np.sum(p2)
            self.omwu_list3.append(p1[self.record])

            if b_count == 20:
                p1 = p1_b.copy()
                p2 = p2_b.copy()
                b_count = 0



    # def run_flbr_b(self):
    #
    #     self.b0 = np.zeros(self.num_actions)
    #     self.b1 = np.zeros(self.num_actions)
    #     self.b0_store = np.zeros(self.num_actions)
    #     self.b1_store = np.zeros(self.num_actions)
    #     # Initialize the strategy profile
    #     p1 = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     p2 = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     # p1[0] -= 0.11
    #     # p1[2] += 0.11
    #
    #     p1_hat = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     p2_hat = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     last_payoffs1 = 0.0
    #     last_payoffs2 = 0.0
    #     first_it = None
    #     xi = 100.0
    #     b_count = 1
    #     # Run the Multiplicative Weights Update algorithm for T iterations
    #
    #     for t in range(self.T):
    #         # Calculate the payoffs for each player
    #         payoffs1 = np.dot(self.game, p2)
    #         payoffs2 = -np.dot(self.game.T, p1)
    #         if t == self.T-1:
    #             print("PAYOFFS1:", sum(payoffs1 * p1))
    #             print("PAYOFFS2:", sum(payoffs2 * p2))
    #
    #         self.b0_temp = self.b0.copy()
    #         self.b1_temp = self.b1.copy()
    #         self.b0_temp += xi * payoffs1
    #         self.b1_temp += xi * payoffs2
    #
    #         p1_hat = np.exp(self.b0_temp)
    #         p2_hat = np.exp(self.b1_temp)
    #
    #         # Calculate payoffs with pi_hats
    #         payoffs1_hat = np.dot(self.game, p2_hat)
    #         payoffs2_hat = -np.dot(self.game.T, p1_hat)
    #
    #         self.b0 += self.eta * payoffs1_hat
    #         self.b1 += self.eta * payoffs2_hat
    #
    #         # Calculate pi^t+1
    #         p1 = np.exp(self.b0)
    #         p2 = np.exp(self.b1)
    #
    #         b_count += 1
    #         if b_count >= 10:
    #             self.b0_store += self.eta * payoffs1_hat
    #             self.b1_store += self.eta * payoffs2_hat
    #
    #         # last_payoffs1 = payoffs1.copy()
    #         # last_payoffs2 = payoffs2.copy()
    #
    #
    #         if b_count == 20:
    #             self.b0 = self.b0_store.copy()
    #             self.b1 = self.b1_store.copy()
    #             b_count = 1
    #
    #
    #         # Normalize the strategy profile
    #         p1 = p1 / np.sum(p1)
    #         p2 = p2 / np.sum(p2)
    #         self.omwu_list3.append(p1[self.record])

        # Print the final strategy profile and expected payoffs for each player
        # print("Final strategy profile:")
        # print("Player 1:", p1)
        # print("Player 2:", p2)
        # print("Expected payoffs:")
        # print("Player 1:", np.dot(payoffs1, p1))
        # print("Player 2:", np.dot(payoffs2, p2))



    # def run_flbr_opt(self):
    #
    #
    #     p1 = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     p2 = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     p1_hat = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #     p2_hat = np.array([1.0 / self.num_actions for _ in range(self.num_actions)])
    #
    #     payoffs1_prev = None
    #     payoffs2_prev = None
    #     first_it = None
    #     xi = 5.0
    #     # Run the Multiplicative Weights Update algorithm for T iterations
    #     for t in range(self.T):
    #         # Calculate the payoffs for each player
    #         payoffs1 = np.dot(self.game, p2)
    #         payoffs2 = -np.dot(self.game.T, p1)
    #
    #         # Calculate pi_hats
    #         p1_hat = p1 * np.exp(xi * payoffs1)
    #         p2_hat = p2 * np.exp(xi * payoffs2)
    #
    #         # Calculate payoffs with pi_hats
    #         payoffs1_hat = np.dot(self.game, p2_hat)
    #         payoffs2_hat = -np.dot(self.game.T, p1_hat)
    #
    #         # Calculate pi^t+1
    #         if first_it == None:
    #             opt_part1 = 2.0 * self.eta * payoffs1_hat
    #             opt_part2 = 2.0 * self.eta * payoffs2_hat
    #             p1 = p1 * np.exp(opt_part1)
    #             p2 = p2 * np.exp(opt_part2)
    #             first_it = 1.0
    #         else:
    #             opt_part1 = 2.0 * self.eta * payoffs1_hat
    #             opt_part2 = 2.0 * self.eta * payoffs2_hat
    #             minus_part1 = 1.0 * self.eta * payoffs1_hat_prev
    #             minus_part2 = 1.0 * self.eta * payoffs2_hat_prev
    #             p1 = p1 * np.exp(opt_part1 - minus_part1)
    #             p2 = p2 * np.exp(opt_part2 - minus_part2)
    #         payoffs1_hat_prev = payoffs1_hat
    #         payoffs2_hat_prev = payoffs2_hat
    #
    #         # Normalize the strategy profile
    #         p1 = p1 / np.sum(p1)
    #         p2 = p2 / np.sum(p2)
    #         self.flbr_opt_list.append(p1[0])

        # Print the final strategy profile and expected payoffs for each player
        # print("Final strategy profile:")
        # print("Player 1:", p1)
        # print("Player 2:", p2)
        # print("Expected payoffs:")
        # print("Player 1:", np.dot(payoffs1, p1))
        # print("Player 2:", np.dot(payoffs2, p2))
        # print("Avg: ", avg / T)

T=1150000
runner = algs(T=T, eta=0.1)
print("OMWU")
runner.run_omwu()
print(runner.p10)
print("OMWU_b")
runner.run_omwu_b()
print(runner.p10)
# print("FLBR")
# runner.run_flbr()
# print("FLBR_b")
# runner.run_flbr_b()
runner.plot_graph()
print("----------")

# e = runner.payoffs1 * runner.p1 * runner.p2 # + runner.payoffs2 * runner.p2
# print("GAME VALUE")
# print(e, sum(e))
# print(runner.payoffs1)
# print(runner.payoffs2)
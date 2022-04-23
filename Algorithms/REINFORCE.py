from Algorithms.PolicyBased import PolicyBased as PB
class REINFORCE(PB):

    def __init__(self, env, model, optimizer, epochs=10, M=5, T=500, gamma=0.9, sigma=None):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.M = M
        self.T = T
        self.gamma = gamma
        self.sigma = sigma


    def epoch(self):
        loss = 0 # initialize the epoch gradient to 0
        reward = 0
        for _ in range(self.M):
            s = self.env.reset()
            h0 = self.sample_trace(s)
            R = 0
            for t in range(len(h0) - 1, -1, -1):
                R = h0[t][2] + self.gamma * R 
                loss += R * h0[t][3]
                reward += h0[t][2]
        
        # compute the epoch's gradient and update weights  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()/self.M, reward/self.M

    
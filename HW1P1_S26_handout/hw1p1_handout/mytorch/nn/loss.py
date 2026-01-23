import numpy as np
import mytorch.nn.activation as ac


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (A - Y) * (A - Y)  # TODO
        sse = np.sum(np.sum(se, axis = 1), axis = 0)  # TODO
        mse = sse / (self.N * self.C)  # TODO
        return mse
        #raise NotImplemented  # TODO - What should be the return value?

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        dLdA = 2 * ((self.A - self.Y) / (self.N * self.C))
        return dLdA
        #raise NotImplemented  # TODO - What should be the return value?


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        Hint: Read the writeup to determine the shapes of all the variables.
        Note: Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]   # TODO

        Ones_C = np.ones((self.C,1))  # TODO
        Ones_N = np.ones((self.N,1))  # TODO

        softInstance = ac.Softmax()
        self.softmax = ac.Softmax().forward(A)  # TODO - Can you reuse your own softmax here, if not rewrite the softmax forward logic?

        #crossentropy = (-Y * np.log(self.softmax)) * Ones_C # TODO
        crossentropy = np.sum((-Y * np.log(self.softmax)) , axis = 1)
        sum_crossentropy_loss = np.sum(crossentropy, axis = 0)  # TODO
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        #raise NotImplemented  # TODO - What should be the return value?
        return mean_crossentropy_loss

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        """
        dLdA = (self.softmax - self.Y) / self.N # TODO
        #raise NotImplemented  # TODO - What should be the return value?
        return dLdA

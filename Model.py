import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator
from tqdm import tqdm

class M_BFGS_SRGPIN_SVM(BaseEstimator):
    def __init__(self, tau1=0.9, tau2=0.9, epsilon1=0.1, epsilon2=0.1, lambd=0.1, eta=100, mu=0.1, C=1, max_iteration=100):
        self.C = C
        self.tau1 = tau1
        self.tau2 = tau2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.lambd = lambd
        self.eta = eta
        self.mu = mu
        self.max_iteration = max_iteration

    def srgp_loss(self, u):
        if u >= (self.epsilon1 / self.tau1) + self.tau1 * self.mu:
            return self.eta * (1 - np.exp(- (self.tau1 * (u - self.epsilon1 / self.tau1) - ((self.tau1 ** 2)* self.mu) / 2) / self.lambd))
        elif self.epsilon1 / self.tau1 <= u <= (self.epsilon1 / self.tau1) + self.tau1 * self.mu:
            return self.eta * (1 - np.exp(- (1 / (2 * self.mu) * (u - self.epsilon1 / self.tau1) ** 2) / self.lambd))
        elif -self.epsilon2 / self.tau2 <= u <= self.epsilon1 / self.tau1:
            return 0
        elif -(self.epsilon2 / self.tau2) - self.tau2 * self.mu <= u <= -self.epsilon2 / self.tau2:
            return self.eta * (1 - np.exp(- (1 / (2 * self.mu) * (u + self.epsilon2 / self.tau2) ** 2) / self.lambd))
        elif u <= -(self.epsilon2 / self.tau2) - self.tau2 * self.mu:
            return self.eta * (1 - np.exp(- (-self.tau2 * (u + self.epsilon2 / self.tau2) - ((self.tau2 ** 2) * self.mu) / 2) / self.lambd))
        else:
            return 0

    def F(self, omega, X, y):
        regularization_term = 0.5 * (np.linalg.norm(omega) ** 2)
        srgp_loss_term = 0
        for i in range(len(y)):
            u = 1 - y[i] * (np.dot(omega, X[i]))
            srgp_loss_term += self.C * self.srgp_loss(u)
        return regularization_term + srgp_loss_term

    def nabla_L_srgp(self, u):
        if self.epsilon1 / self.tau1 + self.tau1 * self.mu <= u:
            return (self.tau1 * self.eta / self.lambd) * np.exp(-((self.tau1 * (u - self.epsilon1 / self.tau1) - self.tau1**2 / (2 * self.mu)) / self.lambd))
        elif self.epsilon1 / self.tau1 <= u <= self.epsilon1 / self.tau1 + self.tau1 * self.mu:
            return (self.eta / (self.lambd * self.mu)) * np.exp(-(1 / (2 * self.mu) * (u - self.epsilon1 / self.tau1)**2) / self.lambd)
        elif -self.epsilon2 / self.tau2 <= u <= self.epsilon1 / self.tau1:
            return 0
        elif (-self.epsilon2 / self.tau2) - self.tau2 * self.mu <= u <= -self.epsilon2 / self.tau2:
            return (self.eta / (self.lambd * self.mu)) * np.exp(-(1 / (2 * self.mu) * (u + self.epsilon2 / self.tau2)**2) / self.lambd)
        elif u <= -self.epsilon2 / self.tau2 - self.tau2 * self.mu:
            return -(self.tau2 * self.eta / self.lambd) * np.exp(-(-self.tau2 * (u + self.epsilon2 / self.tau2) - self.tau2**2 / (2 * self.mu)) / self.lambd)
        else:
            return 0

    def gradient(self, omega, X, y):
        m = X.shape[0]
        n = X.shape[1]
        g_F = omega.copy()
        for i in range(m):
            u = 1 - y[i] * (np.dot(omega, X[i]))
            g_L_srgp = self.nabla_L_srgp(u)
            g_F += -self.C * g_L_srgp * y[i] * X[i]
        return m, n, g_F

    def Update_BFGS(self, B, dw, dg):
        dg_t = dg[:, np.newaxis]
        Bdw = np.dot(B, dw)
        dw_t_B = np.dot(dw, B)
        dwBdw = np.dot(np.dot(dw, B), dw)
        p = dg_t * dg
        u = Bdw[:, np.newaxis] * dw_t_B
        B_new = B + (p / np.dot(dg, dw)) - (u / dwBdw)
        return p, u, B_new

    def armijo_rule(self, omega, H, g, X, y):
        lambda_k = 100
        c = 0.1
        while True:
            lhs = self.F(omega - lambda_k * np.dot(H, g), X, y)
            rhs = self.F(omega, X, y) - c * lambda_k * np.dot(g, np.dot(H, g))
            if lhs <= rhs:
                break
            lambda_k *= 0.5
        return lambda_k

    def fit(self, X, y):
        k = 0
        j = 1
        ones_vector = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones_vector))
        omega = np.zeros(len(X[0]))
        obj_M_BFGS_SRGPIN_SVM = []
        iter_M_BFGS_SRGPIN_SVM = []
        B = np.identity(len(X[0]))
        pbar = tqdm(total=self.max_iteration, desc="Training M_BFGS_SRGPIN_SVM ", leave=False)
        for iteration in range(self.max_iteration):
            iter_M_BFGS_SRGPIN_SVM.append(k)
            cost = self.F(omega, X, y)
            obj_M_BFGS_SRGPIN_SVM.append(cost)
            _, _, g = self.gradient(omega, X, y)
            H = np.linalg.inv(B)
            lambda_k = self.armijo_rule(omega, H, g, X, y)
            omega_new = omega - (lambda_k * (np.matmul(H, g)))
            _, _, g_new = self.gradient(omega_new, X, y)
            dw = omega_new - omega
            if np.all(omega_new == omega) or np.linalg.norm(g_new) < 0.0001 or (np.linalg.norm(dw)**2) == 0:
                print('Stop')
                break
            dg = g_new - g
            if (np.matmul(dg, dw) / (np.linalg.norm(dw)**2)) >= (np.linalg.norm(g)):
                _, _, B_new = self.Update_BFGS(B, dw, dg)
            else:
                B_new = B
            B = B_new
            omega = omega_new
            pbar.update(1)
        self.omega = omega
        self.w = omega[:-1]
        self.b = omega[-1]
        return self.w, self.b

    def predict(self, X):
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions

    def score(self, X, y):
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        accuracy = (y_pred == y).mean()
        return accuracy


    
class M_BFGS_SGPIN_SVM(BaseEstimator):
    def __init__(self, tau1=0.1, tau2=0.1, epsilon1=0.1, epsilon2=0.1, mu=0.1, C=1, max_iteration=100):
        self.C = C
        self.tau1 = tau1
        self.tau2 = tau2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.mu = mu
        self.max_iteration = max_iteration

    def sgp_loss(self, u):
        if u >= (self.epsilon1 / self.tau1) + self.tau1 * self.mu:
            return (self.tau1 * (u - self.epsilon1 / self.tau1) - ((self.tau1 ** 2) * self.mu) / 2)
        elif self.epsilon1 / self.tau1 <= u <= (self.epsilon1 / self.tau1) + self.tau1 * self.mu:
            return (1 / (2 * self.mu) * (u - self.epsilon1 / self.tau1) ** 2)
        elif -self.epsilon2 / self.tau2 <= u <= self.epsilon1 / self.tau1:
            return 0
        elif -(self.epsilon2 / self.tau2) - self.tau2 * self.mu <= u <= -self.epsilon2 / self.tau2:
            return (1 / (2 * self.mu) * (u + self.epsilon2 / self.tau2) ** 2)
        elif u <= -(self.epsilon2 / self.tau2) - self.tau2 * self.mu:
            return (-self.tau2 * (u + self.epsilon2 / self.tau2) - ((self.tau2 ** 2) * self.mu) / 2)
        else:
            return 0

    def F(self, omega, X, y):
        regularization_term = 0.5 * np.linalg.norm(omega) ** 2
        sgp_loss_term = 0
        for i in range(len(y)):
            u = 1 - y[i] * (np.dot(omega, X[i]))
            sgp_loss_term += self.C * self.sgp_loss(u)
        return regularization_term + sgp_loss_term

    def nabla_L_sgp(self, u):
        if self.epsilon1 / self.tau1 + self.tau1 * self.mu <= u:
            return self.tau1
        elif self.epsilon1 / self.tau1 <= u <= self.epsilon1 / self.tau1 + self.tau1 * self.mu:
            return (1 / self.mu) * (u - (self.epsilon1 / self.tau1))
        elif -self.epsilon2 / self.tau2 <= u <= self.epsilon1 / self.tau1:
            return 0
        elif (-self.epsilon2 / self.tau2) - self.tau2 * self.mu <= u <= -self.epsilon2 / self.tau2:
            return (1 / self.mu) * (u + (self.epsilon2 / self.tau2))
        elif u <= -self.epsilon2 / self.tau2 - self.tau2 * self.mu:
            return -self.tau2
        else:
            return 0

    def gradient(self, omega, X, y):
        m = X.shape[0]
        n = X.shape[1]
        g_F = omega.copy()
        for i in range(m):
            u = 1 - y[i] * (np.dot(omega, X[i]))
            g_L_srgp = self.nabla_L_sgp(u)
            g_F += -self.C * g_L_srgp * y[i] * X[i]
        return m, n, g_F

    def Update_BFGS(self, B, dw, dg):
        dg_t = dg[:, np.newaxis]
        Bdw = np.dot(B, dw)
        Bdw = np.dot(B, dw)
        dw_t_B = np.dot(dw, B)
        dwBdw = np.dot(np.dot(dw, B), dw)
        p = dg_t * dg
        u = Bdw[:, np.newaxis] * dw_t_B
        B_new = B + (p / np.dot(dg, dw)) - (u / dwBdw)
        return p, u, B_new

    def armijo_rule(self, omega, H, g, X, y):
        lambda_k = 100
        c = 0.1
        while True:
            lhs = self.F(omega - lambda_k * np.dot(H, g), X, y)
            rhs = self.F(omega, X, y) - c * lambda_k * np.dot(g, np.dot(H, g))
            if lhs <= rhs:
                break
            lambda_k *= 0.5
        return lambda_k

    def fit(self, X, y):
        k = 0
        j = 1
        ones_vector = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones_vector))
        omega = np.zeros(len(X[0]))
        obj_M_BFGS_SGPIN_SVM = []
        iter_M_BFGS_SGPIN_SVM = []
        B = np.identity(len(X[0]))
        pbar = tqdm(total=self.max_iteration, desc="Training M_BFGS_SGPIN_SVM ", leave=False)
        for iteration in range(self.max_iteration):
            k = k + 1
            iter_M_BFGS_SGPIN_SVM.append(k)
            cost = self.F(omega, X, y)
            obj_M_BFGS_SGPIN_SVM.append(cost)
            _, _, g = self.gradient(omega, X, y)
            H = np.linalg.inv(B)
            lambda_k = self.armijo_rule(omega, H, g, X, y)
            omega_new = omega - lambda_k * np.matmul(H, g)
            _, _, g_new = self.gradient(omega_new, X, y)
            dw = omega_new - omega
            omega_new = omega - (lambda_k * (np.matmul(H, g)))
            if np.all(omega_new == omega) or np.linalg.norm(g_new) < 0.0001 or (np.linalg.norm(dw) ** 2) == 0:
                print('Stop')
                break
            dg = g_new - g
            if (np.matmul(dg, dw) / (np.linalg.norm(dw) ** 2)) >= (np.linalg.norm(g)):
                _, _, B_new = self.Update_BFGS(B, dw, dg)
            else:
                B_new = B
            B = B_new
            omega = omega_new
            pbar.update(1)
        pbar.close()
        self.omega = omega
        self.w = omega[:-1]
        self.b = omega[-1]
        return self.w, self.b

    def predict(self, X):
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions

    def score(self, X, y):
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        accuracy = (y_pred == y).mean()
        return accuracy


   
class M_BFGS_SPIN_SVM(BaseEstimator):
    def __init__(self, tau=0.1, mu=0.1, C=1, max_iteration=100):
        self.C = C
        self.tau = tau
        self.mu = mu
        self.max_iteration = max_iteration

    def sp_loss(self, u):
        if u >= 0:
            return (u) - ((self.mu) / (2))
        elif 0 <= u <= self.mu:
            return (1 / (2 * self.mu) * (u) ** 2)
        elif -self.tau * self.mu <= u <= 0:
            return (1 / (2 * self.mu) * (u) ** 2)
        elif u <= -self.tau * self.mu:
            return (-self.tau * (u) - ((self.tau ** 2) * self.mu) / (2))
        else:
            return 0

    def F(self, omega, X, y):
        regularization_term = 0.5 * np.linalg.norm(omega) ** 2
        srgp_loss_term = 0

        for i in range(len(y)):
            u = 1 - y[i] * (np.dot(omega, X[i]))
            srgp_loss_term += self.C * self.sp_loss(u)

        return regularization_term + srgp_loss_term

    def nabla_L_sp(self, u):
        if self.mu <= u:
            return 1
        elif 0 <= u <= self.mu:
            return (1 / self.mu) * (u)
        elif -self.tau * self.mu <= u <= 0:
            return (1 / self.mu) * (u)
        elif u <= -self.tau * self.mu:
            return -self.tau
        else:
            return 0

    def gradient(self, omega, X, y):
        m = X.shape[0]
        n = X.shape[1]
        g_F = omega.copy()
        for i in range(m):
            u = 1 - y[i] * (np.dot(omega, X[i]))
            g_L_srgp = self.nabla_L_sp(u)
            g_F += -self.C * g_L_srgp * y[i] * X[i]
        return m, n, g_F

    def Update_BFGS(self, B, dw, dg):
        dg_t = dg[:, np.newaxis]

        Bdw = np.dot(B, dw)
        dw_t_B = np.dot(dw, B)
        dwBdw = np.dot(np.dot(dw, B), dw)

        p = dg_t * dg
        u = Bdw[:, np.newaxis] * dw_t_B

        B_new = B + (p / np.dot(dg, dw)) - (u / dwBdw)
        return p, u, B_new

    def armijo_rule(self, omega, H, g, X, y):
        lambda_k = 100
        c = 0.1

        while True:
            lhs = self.F(omega - lambda_k * np.dot(H, g), X, y)
            rhs = self.F(omega, X, y) - c * lambda_k * np.dot(g, np.dot(H, g))

            if lhs <= rhs:
                break

            lambda_k *= 0.5
        return lambda_k

    def fit(self, X, y):
        k = 0
        j = 1
        ones_vector = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones_vector))
        omega = np.zeros(len(X[0]))

        obj_M_BFGS_SPIN_SVM = []
        iter_M_BFGS_SPIN_SVM = []

        B = np.identity(len(X[0]))

        pbar = tqdm(total=self.max_iteration, desc="Training M_BFGS_SPIN_SVM ", leave=False)

        for iteration in range(self.max_iteration):
            k = k + 1

            iter_M_BFGS_SPIN_SVM.append(k)

            cost = self.F(omega, X, y)
            obj_M_BFGS_SPIN_SVM.append(cost)
            _, _, g = self.gradient(omega, X, y)

            H = np.linalg.inv(B)

            lambda_k = self.armijo_rule(omega, H, g, X, y)
            omega_new = omega - lambda_k * np.matmul(H, g)

            _, _, g_new = self.gradient(omega_new, X, y)

            dw = omega_new - omega

            omega_new = omega - (lambda_k * (np.matmul(H, g)))

            if np.all(omega_new == omega) or np.linalg.norm(g_new) < 0.0001 or (np.linalg.norm(dw) ** 2) == 0:
                print(np.all(omega_new == omega))
                break

            dg = g_new - g

            if (np.matmul(dg, dw) / (np.linalg.norm(dw) ** 2)) >= (np.linalg.norm(g)):
                _, _, B_new = self.Update_BFGS(B, dw, dg)
            else:
                B_new = B

            B = B_new
            omega = omega_new
            pbar.update(1)

        pbar.close()
        self.omega = omega
        self.w = omega[:-1]
        self.b = omega[-1]
        return self.w, self.b

    def predict(self, X):
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions

    def score(self, X, y):
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        accuracy = (y_pred == y).mean()
        return accuracy
class M_BFGS_SH_SVM(BaseEstimator):
    def __init__(self,  mu = 0.1,C=1, max_iteration = 100):
        self.C = C
        self.mu = mu
        self.max_iteration = max_iteration

    def sh_loss(self, u):
        if u >= 0:
            return (u ) - (( self.mu) / (2 ))
        elif 0 <= u <=   self.mu:
            return (1 / (2 * self.mu) * (u ) ** 2)
        else:
            return 0

    def F(self, omega, X, y):
        regularization_term = 0.5 * np.linalg.norm(omega) ** 2
        sh_loss_term = 0
        
        for i in range(len(y)):
            u = 1 - y[i] * (np.dot(omega, X[i]))
            sh_loss_term += self.C * self.sh_loss(u)

        return regularization_term + sh_loss_term

    def nabla_L_sh(self, u):
        if    self.mu <= u:
            return 1
        elif 0 <= u <= self.mu:
            return (1 / self.mu)*(u )
        else:
            return 0
        
    def gradient(self, omega, X, y):
        m = X.shape[0]
        n = X.shape[1]
        g_F = omega.copy()
        for i in range(m):
            u = 1 - y[i] * (np.dot(omega, X[i]) )
            g_L_sh = self.nabla_L_sh(u)
            g_F += -self.C * g_L_sh * y[i] * X[i]
        return m, n, g_F

    def Update_BFGS(self, B, dw, dg):
        dg_t =  dg[:, np.newaxis]
        Bdw = np.dot(B, dw)
        dw_t_B = np.dot(dw, B)
        dwBdw = np.dot(np.dot(dw, B), dw)
        p = dg_t * dg
        u = Bdw[:, np.newaxis] * dw_t_B
        B_new = B + (p / np.dot(dg, dw)) - (u / dwBdw)
        return p, u, B_new
    
    def armijo_rule(self, omega, H, g, X, y):
        lambda_k = 100
        c=0.1
        
        while True:
            lhs = self.F(omega -  lambda_k * np.dot(H, g), X, y)
            rhs = self.F(omega, X, y) - c * lambda_k * np.dot(g, np.dot(H, g))
            if lhs <= rhs:
                break
            lambda_k *= 0.5
        return lambda_k
    
    def fit(self, X, y):
        k = 0
        j = 1
        ones_vector = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones_vector))
        omega = np.zeros(len(X[0])) 

        obj_M_BFGS_SH_SVM = []
        iter_M_BFGS_SH_SVM = []  

        B = np.identity(len(X[0]))

        pbar = tqdm(total=self.max_iteration, desc="Training M_BFGS_SH_SVM ", leave=False)

        for iteration in range(self.max_iteration):
            k = k + 1
            iter_M_BFGS_SH_SVM.append(k)
            cost = self.F(omega, X, y)
            obj_M_BFGS_SH_SVM.append(cost)
            _, _, g = self.gradient(omega, X, y)

            H = np.linalg.inv(B)
        
            lambda_k =self.armijo_rule(omega,H, g,X, y)
            omega_new = omega - lambda_k * np.matmul(H, g)
            
            _, _, g_new = self.gradient(omega_new, X, y)

            dw = omega_new - omega 
            omega_new = omega - (lambda_k * (np.matmul(H, g)))
            if np.all(omega_new == omega) or np.linalg.norm(g_new) < 0.0001 or (np.linalg.norm(dw)**2)==0:
                print(np.all(omega_new == omega))
                break

            dg = g_new - g 
            if  (np.matmul(dg, dw) / (np.linalg.norm(dw)**2)) >= (np.linalg.norm(g)):
                _, _, B_new = self.Update_BFGS(B, dw, dg)
            else:
                B_new = B

            B = B_new
            omega = omega_new
            pbar.update(1)

        pbar.close()
        self.omega = omega
        self.w = omega[:-1]   
        self.b = omega[-1]     
        return self.w, self.b
    
    def predict(self, X):
        decision = np.dot(X, self.w) + self.b
        predictions = np.sign(decision)
        predictions = np.where(predictions == 0, -1, predictions)
        return predictions
    
    def score(self, X, y):
        y_pred = np.dot(X, self.w) + self.b
        y_pred = np.sign(y_pred)
        y_pred = np.where(y_pred == 0, -1, y_pred)
        accuracy = (y_pred == y).mean()
        return accuracy

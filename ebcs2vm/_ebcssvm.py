import numpy as np
from sklearn.utils import check_random_state 
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels    
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from randomgen import Generator, DSFMT
from .utils import _utils

class EBCSSVM(BaseEstimator):        
    """Evolutionary Bilevel Cost-Sensitive Support Vector Machine.
    
    It implements EBCS-SVM [1] to train Support Vector Machines for Imbalanced
    Classification Problems via Evolutionary bilevel optimization of the
    costs for each class.
    
    It currently supports binary classification
    
    Parameters
    ----------
    upper_evaluations : int, default=1000
        The number of fitness function evaluation for the upper level problem.
    upper_population : int, default=30
        The number of individuals for the upper level problem.
    lower_evaluations : int, default=1000
        The number of iterations for SMO in the lower level problem.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    costs_ : ndarray of shape (n_classes,)
        The learned cost for each class.
    gamma_ : float,
        The gamma value for RBF kernel.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.
    is_fitted_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)
    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.
    support_ : ndarray of shape (n_SV)
        Indices of support vectors.
    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.
    obj_upper_ : float,
        BER reached by the best solution at the upper level.
    obj_lower_ : float,
        Objective function value at the lower level for the best solution.
    KKT_condition_ : float,
        KKT value for the lower level problem.
    References
    ----------
    .. [1] `Rosales-Pérez, Alejandro, García, Salvador, and Herrera, Francisco
        (2022). "Handling Imbalanced Classification Problems With Support Vector 
        Machines via Evolutionary Bilevel Optimization." IEEE Transactions on
        Cybernetics.
        < https://doi.org/10.1109/TCYB.2022.3163974>`_
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from ebcs2vm import EBCSSVM
    >>> clf = make_pipeline(StandardScaler(), EBCSSVM())
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))])
    >>> print(clf.predict([[-0.8, -1]]))
    """
    def __init__(self,upper_evaluations=1000, upper_population=30,
                 lower_evaluations=1000, random_state=None):
        self.upper_evaluations = upper_evaluations
        self.upper_population = upper_population
        self.lower_evaluations = lower_evaluations
        self.random_state = random_state
    
    def __reproduce(self, pop, dim, fitness, pop_size, mCR, mF, pmin):
        idx_best = np.argsort(fitness)
        idx_all = np.argsort(self.random_state.uniform(size = (pop_size, pop_size)), 
                             axis = 1)[:,:2]
        idx_1 = idx_all[:, 0]
        idx_2 = idx_all[:, 1]
        r = self.random_state.randint(0, pop_size-1, (pop_size))
        CR = np.minimum(1, np.maximum(0, self.random_state.normal(mCR[r],0.1)))
        p = (0.2 - pmin) * (self.random_state.uniform(size=(pop_size))) + pmin

        idx_p = idx_best[self.random_state.randint(np.int32(pop_size * p), size = (pop_size))]
        F = self.random_state.standard_cauchy((pop_size)) * 0.1 + mF[r]
        if np.any(F < 0):
            tmp = self.random_state.standard_cauchy((pop_size)) * 0.1 + mF[r]
            F[F < 0] = tmp[F < 0]
            F[F < 0.001] = 0.001
        F = np.minimum(1, F)
        trial = pop + np.multiply(F[:, None], pop[idx_p,:] - pop) + \
            np.multiply(F[:, None], pop[idx_1,:] - pop[idx_2,:])
        trial = trial
        mask = (self.random_state.uniform(size=(pop_size, dim))) < CR[:, None]
        mask[np.arange(pop_size), self.random_state.randint(dim, size = (pop_size))] = False
        trial[mask] = pop[mask]
        
        trial = np.maximum([-5, -5, -10], 
                            np.minimum(trial, [10, 10, 5]))  
        return trial, CR, F
    
    def __initilizeDEParameters(self, pop_size):
        MCR = 0.5 * np.ones((pop_size))
        MF = 0.5 * np.ones((pop_size))
        k = 0;
        pmin = 2 / pop_size
        t = np.ceil(pop_size / 2)
        
        return MCR, MF, k, pmin, t
    
    def __updateDEParameters(self, improvement, CR, F, deltaFit, MCR, MF, k, pop_size):
        SCR = CR[improvement]
        SF = F[improvement]
        wCR = deltaFit[improvement] / (deltaFit[improvement]).sum()
        meanWCR = np.dot(wCR, SCR)
        meanLF = (SF**2).sum() / SF.sum()
        MCR[k] = meanWCR
        MF[k] = meanLF
        k = np.mod(k + 1, pop_size)
        
        return MCR, MF, k
    
    def __trainSVM(self, lower_pop_, kernel_mat_, y_, lower_evaluations, box_constraints):
        support_vectors = lower_pop_ > 0
        lower_pop = np.zeros((support_vectors.sum()))
        box = box_constraints[support_vectors]
        y = y_[support_vectors]
        Q = np.multiply(np.dot(y[:, None], y[None, :]), kernel_mat_[support_vectors][:, support_vectors])
        fit = np.zeros((1))
        b_ = np.zeros((1))
        p = np.ones(y.shape)
        is_positive = y == 1
        p[y > 0] = 2 * box[is_positive][0] / (box[is_positive][0] + box[~is_positive][0])
        p[y < 0] = 2 * box[~is_positive][0] / (box[is_positive][0] + box[~is_positive][0])
        _utils.smo(Q, y.astype(int), lower_pop, p, b_, box, lower_evaluations, fit)
        
        out = np.zeros((lower_pop_.shape[0]))
        out[support_vectors] = lower_pop
        return out, fit, b_
    
    def __evalHyperParameters(self, lower, kernel_mat, y, box_constraints, b_):
        alpha_y = np.multiply(lower, y)
        y_pred = np.dot(kernel_mat, alpha_y)
        is_positive = y == 1
        
        y_hat = (y_pred - alpha_y) + b_
        is_right = (np.multiply(y_hat, y)) > 0
        true_positive_rate = (is_right & is_positive).sum(axis = 0)
        true_negative_rate = (is_right & ~is_positive).sum(axis = 0) 
        false_positive_rate = (~is_right & ~is_positive).sum(axis = 0)
        false_negative_rate = (~is_right & is_positive).sum(axis = 0) 
        BER = 0.5 * (np.divide(false_negative_rate, true_positive_rate + 
                               false_negative_rate) +  
                     np.divide(false_positive_rate, true_negative_rate + 
                               false_positive_rate)) 
        
        return BER, b_, (~is_right & (lower == 0))

    def fit(self, X, y):        
        X, y = check_X_y(X, y, accept_sparse=False)
        y = y.astype(int)
                
        upper_evaluations = self.upper_evaluations
        upper_population = self.upper_population
        lower_evaluations = self.lower_evaluations
        # lower_population = self.lower_population 
        
        random_state = check_random_state(self.random_state)
        random_seed = random_state.randint(np.iinfo(np.int32).max)
        self.random_state = Generator(DSFMT(seed = random_seed))
        
        upper_pop = self.random_state.uniform(size=(upper_population, 3))
        lower_bound = np.array([-5, -5, -10])
        upper_bound = np.array([10, 10, 5])
        upper_pop = np.multiply(upper_bound - lower_bound, upper_pop) + lower_bound
        
        population = []
        dst_mat = _utils.squared_Euclidean(X, X)

        # Getting the unique class labels
        labels = unique_labels(y)
        if len(labels) != 2:
            raise ValueError('Only binary classification is supported')
        # Relabeling
        if y[y == labels[0]].shape[0] < y[y == labels[1]].shape[0]:
            labels = labels[::-1]
        idx_n = y == labels[0]
        y[idx_n] = -1
        y[~idx_n] = 1
        
        upper_fitness = []

        # Upper population C1, C2, gamma
        # Lower population alpha's
        count = -1
        lower_ = np.ones((X.shape[0]))
        box_ = np.ones(y.shape)
        for upper_individual in upper_pop:
            count += 1
            box_[~idx_n] = 2**upper_individual[1]
            box_[idx_n] = 2**upper_individual[0]
            kernel_matrix_tmp = _utils.rbf_kernel_2(dst_mat, 2**upper_individual[2])
            lower_pop, lowerFitness, b_ = self.__trainSVM(lower_, kernel_matrix_tmp, y,  
                                                      lower_evaluations, box_)
            uFitness, b, err = self.__evalHyperParameters(lower_pop, 
                                                      kernel_matrix_tmp, 
                                                      y, box_, b_)
            upper_fitness = np.append(upper_fitness, uFitness)
            popTmp = {'upper_pop':upper_individual, 'lower_pop':lower_pop,
                      'upper_fitness':uFitness,\
                      'lowerFitness': lowerFitness, 'b':b, 'err': err}
            population.append(popTmp)

        # Evolutionary 
        MCR, MF, k, pmin, t = self.__initilizeDEParameters(upper_population)
        improvement = np.zeros((upper_population,),dtype=bool)
        deltaFit = np.zeros((upper_population,))
        evals = len(upper_fitness)
        stop = evals > upper_evaluations
        best_ = np.min(upper_fitness)
        nn_num = int(upper_population / 2)
        prob_nn = np.ones((nn_num))
        prob_nn /= prob_nn.sum()
        list_nn = np.arange(nn_num) + 1
        while not stop:
            succ_nn = np.zeros((nn_num))
            Neighbors=NearestNeighbors(n_neighbors=int(0.5 * upper_population))
            Neighbors.fit(np.divide(upper_pop - lower_bound, upper_bound - lower_bound))
            trial, CR, F = self.__reproduce(upper_pop, 3, upper_fitness, upper_population, MCR, MF, pmin)
            _, idx_neighbors = Neighbors.kneighbors(np.divide(trial - lower_bound, upper_bound - lower_bound))
            for count, upper_individual in enumerate(trial):
                evals += 1
                box_[~idx_n] = 2**upper_individual[1]
                box_[idx_n] = 2**upper_individual[0]
                kernel_matrix_tmp = _utils.rbf_kernel_2(dst_mat, 2**upper_individual[2])
                history_pop = population[idx_neighbors[count, 0]]['lower_pop'][None, :]
                idx_k_nn = np.nonzero(np.cumsum(prob_nn) > self.random_state.uniform())[0][0]
                nn = list_nn[idx_k_nn]
                for idx_neighbor in idx_neighbors[count, 1:nn]:
                    history_pop = np.append(history_pop, 
                                            population[idx_neighbor]['lower_pop'][None, :], 
                                            axis = 0)
                    history_pop[-1, population[idx_neighbor]['err'] > 0] = 1
                lower_pop, lowerFitness, b_ = self.__trainSVM(np.mean(history_pop > 0, axis = 0), 
                                                          kernel_matrix_tmp, 
                                                          y, lower_evaluations, 
                                                          box_)
                uFitness, b, err = self.__evalHyperParameters(lower_pop, 
                                                          kernel_matrix_tmp, 
                                                          y, box_, b_)
                deltaFit[count] = np.abs(uFitness - upper_fitness[count])
                is_better = (upper_fitness[count] - uFitness) > 1e-2
                equivalent = -1e-2 <= (upper_fitness[count] - uFitness) <= 1e-2
                less_support = (lower_pop > 0).sum() < (population[count]['lower_pop'] > 0).sum()
                if is_better or (equivalent and less_support):
                    popTmp = {'upper_pop':upper_individual, 'lower_pop':lower_pop,
                              'upper_fitness':uFitness,
                              'lowerFitness': lowerFitness, 'b':b, 'err': err}
                    population[count] = popTmp
                    upper_fitness[count] = uFitness
                    upper_pop[count] = upper_individual
                    improvement[count] = True
                    succ_nn[idx_k_nn] += 1
                    if uFitness < best_:
                        best_ = uFitness
                else:
                    improvement[count] = False

                if (evals == upper_evaluations) or (best_ < 1e-13): 
                    stop = True
                    break
                
            if np.any(succ_nn > 0):
                prob_nn += (succ_nn / succ_nn.sum())
                prob_nn /= prob_nn.sum()
            if any(improvement):
                MCR, MF, k = self.__updateDEParameters(improvement, CR, F, 
                                                       deltaFit, MCR, MF, k, 
                                                       upper_population)
            convergent = np.std(upper_fitness) < 1e-3
            stop = convergent or (not (evals < upper_evaluations))

        idx_best = np.argsort(upper_fitness)
        best_ = idx_best[0]
        for idx_ in idx_best[1:]:
            if (upper_fitness[idx_] - upper_fitness[idx_best[0]]) <= 0.01:
                if (population[idx_]['lower_pop'] > 0).sum() < (population[best_]['lower_pop'] > 0).sum():
                    best_ = idx_
            else:
                break
        idx_best = best_
        solution = population[idx_best]
        support_vectors = solution['lower_pop']
        self.support_ = np.nonzero(support_vectors > 0)[0]
        self.intercept_ = solution['b']
        self.X_ = X[self.support_]
        self.y_ = y[self.support_]
        self.dual_coef_ = np.multiply(support_vectors[self.support_], self.y_)
        self.n_support_ = np.array([(self.dual_coef_ < 0).sum(), (self.dual_coef_ > 0).sum()])
        self.gamma_ = 2**solution['upper_pop'][2]
        self.costs_ = 2**solution['upper_pop'][0:2]
        self.obj_upper_ = solution['upper_fitness']
        self.obj_lower_ = solution['lowerFitness'][0]
        self.classes_ = labels
        self.KKT_condition_ = self.dual_coef_.sum()
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse = False)
        check_is_fitted(self, 'is_fitted_')
        
        kernel_mat = _utils.rbf_kernel(X, self.X_, self.gamma_)
        yscore = np.dot(kernel_mat, self.dual_coef_) + self.intercept_
        yp = np.sign(yscore)
        
        idx_n = yp < 0
        yp[idx_n] = self.classes_[0]
        yp[~idx_n] = self.classes_[1]
        
        return yp
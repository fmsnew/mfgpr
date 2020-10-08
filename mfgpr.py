"""Gaussian processes regression. """

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.base import MultiOutputMixin
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.exceptions import ConvergenceWarning


class GaussianProcessCoKriging(BaseEstimator, RegressorMixin,
                               MultiOutputMixin):
    """Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:

       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be minimized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Target values in training data (also required for prediction)

    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y) # doctest: +ELLIPSIS
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True) # doctest: +ELLIPSIS
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))

    """
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, random_state=None, rho=0, rho_bounds=(-1, 1),
                 eval_gradient=False):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.rho = rho
        self.rho_bounds = rho_bounds
        self.eval_gradient = eval_gradient

    def fit(self, X_l, y_l, X_h, y_h):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_l_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_l_ = clone(self.kernel)
        self.kernel_d_ = clone(self.kernel_l_)

        self.rng = check_random_state(self.random_state)

        X_l, y_l = check_X_y(X_l, y_l, multi_output=True, y_numeric=True)
        X_h, y_h = check_X_y(X_h, y_h, multi_output=True, y_numeric=True)
        self.n_l_ = len(X_l)

        # Normalize target value
        if self.normalize_y:
            self._y_l_train_mean = np.mean(y_l, axis=0)
            self._y_h_train_mean = np.mean(y_h, axis=0)
            # demean y
            y_l = y_l - self._y_l_train_mean
            y_h = y_h - self._y_h_train_mean
        else:
            self._y_l_train_mean = np.zeros(1)
            self._y_h_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.vstack((X_l, X_h))
        self.y_train_ = np.hstack((y_l, y_h))

        theta_initial = np.hstack((np.array([self.rho]), self.kernel_l_.theta, self.kernel_d_.theta))
        if self.optimizer is not None and self.kernel_l_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=self.eval_gradient):
                if eval_gradient:
                    raise Warning("eval_gradient = True mode is not implemented yet!")
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            theta_bounds = np.r_[np.array(self.rho_bounds)[np.newaxis],
                            self.kernel_l_.bounds,
                            self.kernel_d_.bounds]
            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      theta_initial,
                                                      theta_bounds, 
                                                      self.eval_gradient))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not (np.isfinite(self.kernel_l_.bounds).all() and np.isfinite(self.kernel_d_.bounds).all() and np.isfinite(self.rho_bounds).all()):
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = np.vstack((np.array(self.rho_bounds).reshape(1, -1), self.kernel_l_.bounds, self.kernel_d_.bounds))
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = np.hstack((
                                        self.rng.uniform(bounds[0, 0], bounds[0, 1]),
                                        np.exp(self.rng.uniform(bounds[1:, 0], bounds[1:, 1]))
                                        
                    ))
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds, self.eval_gradient))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            best_hyperparams = optima[np.argmin(lml_values)][0]
            self.rho = best_hyperparams[0]
            self.kernel_l_.theta = best_hyperparams[1:1 + len(self.kernel_l_.theta)]
            self.kernel_d_.theta = best_hyperparams[1 + len(self.kernel_l_.theta):]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(theta_initial)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K_lf = self.kernel_l_(self.X_train_[:self.n_l_])
        K = np.vstack((
                    np.hstack(( self.kernel_l_(self.X_train_[:self.n_l_]), 
                                self.rho * self.kernel_l_(self.X_train_[:self.n_l_], self.X_train_[self.n_l_:]) )),
                    np.hstack(( self.rho * self.kernel_l_(self.X_train_[self.n_l_:], self.X_train_[:self.n_l_]), 
                                self.rho**2 *self.kernel_l_(self.X_train_[self.n_l_:]) + self.kernel_d_(self.X_train_[self.n_l_:]) ))
                     ))
        K_lf[np.diag_indices_from(K_lf)] += self.alpha
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_lf_ = cholesky(K_lf, lower=True)  # Line 2 (lf)
            self.L_ = cholesky(K, lower=True)  # Line 2 
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
            self._K_lf_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator.",) + exc.args
            raise
        self.alpha_lf_ = cho_solve((self.L_lf_, True), y_l)  # Line 3 (Lf)
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self

    def predict_lf(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X = check_array(X)

        if not hasattr(self, "X_train_"):  
            raise Warning("Unfitted GP error. Call fit method first.")
        else:  # Predict based on GP posterior
            K_trans = self.kernel_l_(X, self.X_train_[:self.n_l_])
            y_mean = K_trans.dot(self.alpha_lf_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_l_train_mean + y_mean  # undo normal.
            if return_cov:
                v = cho_solve((self.L_lf_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_l_(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                if self._K_lf_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_lf_.T,
                                             np.eye(self.L_lf_.shape[0]))
                    self._K_lf_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = self.kernel_l_.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_lf_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X = check_array(X)

        if not hasattr(self, "X_train_"):  
            raise Warning("Unfitted GP error. Call fit method first.")
        else:  # Predict based on GP posterior
            K_trans = np.hstack((
                                   self.rho * self.kernel_l_(X, self.X_train_[:self.n_l_]), 
                                   self.rho**2 * self.kernel_l_(X, self.X_train_[self.n_l_:]) + self.kernel_d_(X, self.X_train_[self.n_l_:])
                                  ))
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_h_train_mean + y_mean  # undo normal.
            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.rho**2 * self.kernel_l_(X) + self.kernel_d_(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = self.rho**2 * self.kernel_l_.diag(X) + self.kernel_d_.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel_l = self.kernel_l_.clone_with_theta(theta[1: 1 + len(self.kernel_l_.theta)])
        kernel_d = self.kernel_d_.clone_with_theta(theta[-len(self.kernel_d_.theta):])
        rho = theta[0]

        if eval_gradient:
            raise Warning("eval_gradient = True mode is not implemented yet!")
#             K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = np.vstack((
                            np.hstack(( kernel_l(self.X_train_[:self.n_l_]), 
                                        rho * kernel_l(self.X_train_[:self.n_l_], self.X_train_[self.n_l_:]) )),
                            np.hstack(( rho * kernel_l(self.X_train_[self.n_l_:], self.X_train_[:self.n_l_]), 
                                        rho**2 * kernel_l(self.X_train_[self.n_l_:]) + kernel_d(self.X_train_[self.n_l_:]) ))
                     ))

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum() # -0.5 log det (K) = log(L) (since K = LL^T)
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            raise Warning("eval_gradient = True mode is not implemented yet!")
#             tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
#             tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
#             # Compute "0.5 * trace(tmp.dot(K_gradient))" without
#             # constructing the full matrix tmp.dot(K_gradient) since only
#             # its diagonal is required
#             log_likelihood_gradient_dims = \
#                 0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
#             log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            raise Warning("eval_gradient = True mode is not implemented yet!")
#             return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds, eval_grad = False):
        if self.optimizer == "fmin_l_bfgs_b":
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, approx_grad = 1 - int(eval_grad))
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict,
                              ConvergenceWarning)
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min
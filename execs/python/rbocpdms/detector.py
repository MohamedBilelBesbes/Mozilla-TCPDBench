import numpy as np
import scipy
from scipy import special
import time

from .BVAR_NIG_DPD import BVARNIGDPD
from .BVAR_NIG import BVARNIG


class Detector:
    def __init__(self, data, model_universe, model_prior, cp_model,
                 S1, S2, T, exo_data=None, num_exo_vars=None, threshold=None,
                 store_rl=False, store_mrl=False, trim_type="keep_K",
                 notifications=50,
                 save_performance_indicators=False,
                 training_period=200,
                 generalized_bayes_rld=False,
                 alpha_rld=None,
                 alpha_rld_learning=False,
                 alpha_param_learning=False,
                 alpha_param=None,
                 loss_der_rld_learning=None,
                 loss_param_learning=None,
                 step_size_rld_learning=None,
                 step_size_param_learning=None,
                 eps_param_learning=None,
                 alpha_param_opt_t=100,
                 alpha_rld_opt_t=100):
        self.gradient = 0.0
        self.data = data.reshape(T, S1 * S2)
        self.model_universe = model_universe
        self.model_prior = model_prior
        self.cp_model = cp_model
        if isinstance(model_universe, list):
            self.Q = int(len(model_universe))
        else:
            self.Q = model_universe.shape[0]
        self.T, self.S1, self.S2 = T, S1, S2
        self.threshold = threshold
        self.store_rl, self.store_mrl = store_rl, store_mrl
        self.not_all_initialized = True
        self.m_star_old = 0
        self.first_model_initialized = False
        self.trim_type = trim_type
        self.notifications = notifications
        self.save_performance_indicators = save_performance_indicators
        self.training_period = training_period
        self.negative_log_likelihood = []
        self.negative_log_likelihood_fixed_pars = []
        self.MSE = []
        self.MAE = []
        if exo_data is not None:
            self.exo_data = exo_data.reshape(T, S1 * S2, num_exo_vars)
        else:
            self.exo_data = exo_data
        self.num_exo_vars = num_exo_vars
        self.evidence = -np.inf
        self.MAP = None
        self.y_pred_mean = np.zeros(shape=(self.S1, self.S2))
        self.y_pred_var = np.zeros(shape=(self.S1 * self.S2, self.S1 * self.S2))
        self.model_and_run_length_log_distr = (-np.inf * np.ones(shape=(self.Q, self.T + 1)))
        self.storage_run_length_log_distr = []
        self.storage_model_and_run_length_log_distr = []
        self.storage_all_retained_run_lengths = []
        self.storage_mean = np.zeros(shape=(self.T, self.S1, self.S2))
        self.storage_var = np.zeros((self.T, self.S1 * self.S2))
        self.storage_log_evidence = -np.inf * np.ones(shape=self.T)
        self.log_MAP_storage = np.array([0, 0])
        self.CPs = [[]] * self.T
        self.MAP_segmentation = [np.array([[], []])]
        self.smallest_lag_length = 99999
        for model in self.model_universe:
            if model.has_lags:
                self.smallest_lag_length = min(self.smallest_lag_length, model.lag_length)
            else:
                self.smallest_lag_length = 0
        self.max_lag_length = 0
        for model in self.model_universe:
            if model.has_lags:
                self.max_lag_length = max(self.max_lag_length, model.lag_length)
        self.alpha_param_learning = alpha_param_learning
        if (alpha_param is not None) and alpha_param_learning == "together":
            self.alpha_param = alpha_param
            self.gradient_alpha_param_count = 0
            self.gradient_alpha_param = 0
        elif alpha_param_learning == "together":
            self.alpha_param = self.model_universe[0].alpha_param
            self.gradient_alpha_param_count = 0
            self.gradient_alpha_param = 0
            print("WARNING! You are using DPD for parameter inference " +
                  "and want to optimize alpha_param across all models, " +
                  "but you have not specified an initial value in the " +
                  "detector object. The alpha_param of the first " +
                  "model in the model universe was chosen instead")
        if ((alpha_param_learning == "individual") or
                alpha_param_learning == "together"):
            if alpha_param_learning == "individual":
                self.gradient_alpha_param_count = np.zeros(self.Q)
                self.gradient_alpha_param = np.zeros(self.Q)
            for model in self.model_universe:
                if isinstance(model, BVARNIGDPD):
                    model.alpha_param_learning = True
                    if alpha_param_learning == "together":
                        model.alpha_param = self.alpha_param
        self.alpha_param_opt_t = alpha_param_opt_t
        self.alpha_rld_opt_t = alpha_rld_opt_t
        self.alpha_opt_count = 0
        self.alpha_rld_learning = alpha_rld_learning
        self.generalized_bayes_rld = generalized_bayes_rld
        self.alpha_rld = alpha_rld
        self.alpha_list = []
        if generalized_bayes_rld == "power_divergence":
            for model in self.model_universe:
                model.generalized_bayes_rld = "power_divergence"
                model.alpha_rld = self.alpha_rld
                model.alpha_rld_learning = self.alpha_rld_learning
            self.jlp_scale = None
            self.gradient_alpha_rld_count = 0
            self.gradient_alpha_rld = 0
        self.C = 1.0
        if loss_der_rld_learning is None:
            loss_der_rld_learning = Detector.bounded_absolute_loss_derivative
        elif loss_der_rld_learning == "squared_loss" or loss_der_rld_learning == "squared_loss_derivative":
            loss_der_rld_learning = Detector.squared_loss_derivative
        elif loss_der_rld_learning == "absolute_loss" or loss_der_rld_learning == "absolute_loss_derivative":
            loss_der_rld_learning = Detector.absolute_loss_derivative

        if loss_param_learning is None:
            loss_param_learning = Detector.bounded_absolute_loss
        elif loss_param_learning == "squared_loss":
            loss_param_learning = Detector.squared_loss
        elif loss_param_learning == "absolute_loss":
            loss_param_learning = Detector.absolute_loss

        if step_size_rld_learning is None:
            step_size_rld_learning = Detector.step_size_gen_rld
        if step_size_param_learning is None:
            step_size_param_learning = Detector.step_size_gen_rld
        if eps_param_learning is None:
            eps_param_learning = Detector.eps_gen

        self.loss_der_rld_learning = loss_der_rld_learning
        self.loss_param_learning = loss_param_learning
        self.step_size_rld_learning = step_size_rld_learning
        self.step_size_param_learning = step_size_param_learning
        if step_size_param_learning is None:
            self.step_size_param_learning = (
                Detector.default_step_size_param_learning)
        self.eps_param_learning = eps_param_learning
        self.all_retained_run_lengths = np.array([], dtype=int)

    def reinstantiate(self, new_model_universe):
        data, model_universe = self.data, new_model_universe
        model_prior, cp_model = self.model_prior, self.cp_model
        S1, S2, T = self.S1, self.S2, self.T
        exo_data, num_exo_vars = self.exo_data, self.num_exo_vars
        threshold = self.threshold
        store_rl, store_mrl = self.store_rl, self.store_mrl
        trim_type, notifications = self.trim_type, self.notifications
        save_performance_indicators = self.save_performance_indicators
        training_period = self.training_period
        new_detector = Detector(data, model_universe, model_prior, cp_model,
                                S1, S2, T, exo_data, num_exo_vars, threshold,
                                store_rl, store_mrl, trim_type,
                                notifications,
                                save_performance_indicators,
                                training_period)
        return new_detector

    def run(self, start=None, stop=None):
        if start is None:
            start = 1
        if stop is None:
            stop = self.T
        time_start = time.time()
        for t in range(start - 1, stop - 1):
            self.next_run(self.data[t, :], t + 1)
        self.execution_time = time.time() - time_start

    def next_run(self, y, t):
        if self.max_lag_length + 3 < t and self.alpha_param_opt_t < t:
            if t > 3:
                if self.CPs[t - 2][-1][0] != self.CPs[t - 3][-1][0]:
                    self.alpha_opt_count = self.alpha_opt_count + 1
                    self.update_alpha_param(y, self.alpha_opt_count, True)
                else:
                    self.update_alpha_param(y, self.alpha_opt_count, False)
        self.update_all_joint_log_probabilities(y, t)
        if self.save_performance_indicators and t > self.training_period:
            self.save_negative_log_likelihood(t)
            self.save_MSE(y, t)
        self.update_log_evidence()
        self.trim_run_length_log_distributions(t)
        self.update_run_length_log_distribution(t)
        if not self.not_all_initialized:
            self.prediction_y(y, t)
            self.storage(t)
        if ((not self.not_all_initialized) and
                self.generalized_bayes_rld == "power_divergence" and
                self.alpha_rld_learning and
                self.alpha_rld_opt_t < t):
            if t >= 3:
                if self.CPs[t - 2][-1][0] != self.CPs[t - 3][-1][0]:
                    self.update_alpha_rld(y, self.alpha_opt_count, True)
                else:
                    self.update_alpha_rld(y, self.alpha_opt_count, False)
        if self.not_all_initialized:
            count_total, count_init = 0, 0
            for model in self.model_universe:
                if model.has_lags:
                    if t - model.lag_length >= 1:
                        count_total += 1
                        count_init += 1
                    else:
                        count_total += 1
            if count_total > count_init:
                self.not_all_initialized = True
            else:
                self.not_all_initialized = False
        self.MAP_estimate(t)
        if not self.not_all_initialized:
            self.update_priors(t)

    def rescale_DPD_run_length_log_distribution(self, t):
        if self.jlp_scale is None:
            log_scale_new = (np.max([model.joint_log_probabilities for model in self.model_universe]) - 1)
        else:
            min_ = (np.min([model.joint_log_probabilities for model in self.model_universe]) - 1)
            log_scale_new = min_
        self.jlp_scale = log_scale_new

    def update_alpha_param(self, y, t, update=True):
        if self.alpha_param_learning == "together":
            eps = self.eps_gen(t - 1)
            DPD_model_indices = []
            list_model_log_evidences_p_eps = []
            list_model_log_evidences_m_eps = []
            list_post_mean_p_eps = []
            list_post_mean_m_eps = []
            for (m, model) in zip(range(0, self.Q), self.model_universe):
                if isinstance(model, BVARNIGDPD):
                    list_model_log_evidences_p_eps.append(model.model_log_evidence_p_eps)
                    list_model_log_evidences_m_eps.append(model.model_log_evidence_m_eps)
                    list_post_mean_p_eps.append(model.post_mean_p_eps)
                    list_post_mean_m_eps.append(model.post_mean_m_eps)
                    DPD_model_indices.append(m)
            total_evidence_p_eps = scipy.special.logsumexp(list_model_log_evidences_p_eps)
            total_evidence_m_eps = scipy.special.logsumexp(list_model_log_evidences_m_eps)
            model_posteriors_p_eps = np.exp(np.array(list_model_log_evidences_p_eps) - total_evidence_p_eps)
            model_posteriors_m_eps = np.exp(np.array(list_model_log_evidences_m_eps) - total_evidence_m_eps)
            post_mean_p_eps = (np.array(list_post_mean_p_eps) * model_posteriors_p_eps[:, np.newaxis])
            post_mean_m_eps = (np.array(list_post_mean_m_eps) * model_posteriors_m_eps[:, np.newaxis])
            loss_p_eps = self.loss_param_learning(post_mean_p_eps - y.flatten(), self.C)
            loss_m_eps = self.loss_param_learning(post_mean_m_eps - y.flatten(), self.C)
            self.gradient_alpha_param = (self.gradient_alpha_param + (loss_p_eps - loss_m_eps) / (2 * eps))
            self.gradient_alpha_param_count = (self.gradient_alpha_param_count + 1)
            if update:
                step_size = self.step_size_param_learning(t)
                abs_increment = min(0.1,
                                    step_size * (1.0 / self.gradient_alpha_param_count) * self.gradient_alpha_param)
                self.alpha_param = min(
                    max(pow(10, -10), self.alpha_param - abs_increment * np.sign(self.gradient_alpha_param)),
                    10.0
                )
                self.gradient_alpha_param = 0
                self.gradient_alpha_param_count = 0
                for m in DPD_model_indices:
                    self.model_universe[m].alpha_param = self.alpha_param
                    self.model_universe[m].alpha_param_list.append(self.alpha_param)
        elif self.alpha_param_learning == "individual":
            for (m, model) in zip(range(0, self.Q), self.model_universe):
                if isinstance(model, BVARNIGDPD):
                    loss_p_eps = np.sum(np.abs(model.post_mean_p_eps - y.flatten()))
                    loss_m_eps = np.sum(np.abs(model.post_mean_m_eps - y.flatten()))
                    self.gradient_alpha_param[m] = (
                                self.gradient_alpha_param[m] + (loss_p_eps - loss_m_eps) / (2 * model.eps))
                    self.gradient_alpha_param_count[m] = (self.gradient_alpha_param_count[m] + 1)
                    if update:
                        step_size = self.step_size_param_learning(t)
                        abs_increment = min(0.1,
                                            step_size *
                                            (1.0 / self.gradient_alpha_param_count[m]) *
                                            self.gradient_alpha_param[m])
                        model.alpha_param = min(
                            max(
                                pow(10, -10),
                                model.alpha_param - abs_increment * np.sign(self.gradient_alpha_param[m])
                            ),
                            10.0
                        )
                        model.alpha_param_list.append(model.alpha_param)
                        self.gradient_alpha_param[m] = 0
                        self.gradient_alpha_param_count[m] = 0

    def update_run_length_log_distribution(self, t):
        indices_initialized_models = []
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            if model.has_lags:
                model_lag = model.lag_length
            else:
                model_lag = 0
            if model_lag < t:
                indices_initialized_models.append(m)
        num_initialized = int(len(indices_initialized_models))
        if (not self.first_model_initialized) and num_initialized > 0:
            self.first_model_initialized = True
            self.run_length_log_distr = 0
        elif not self.first_model_initialized:
            return None
        has_both = False
        all_run_lengths = np.array([])
        for model in self.model_universe[indices_initialized_models]:
            all_run_lengths = np.union1d(all_run_lengths, model.retained_run_lengths)
            if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                has_both = True
        if has_both:
            all_run_lengths = np.append(all_run_lengths, t)
        all_run_lengths = all_run_lengths.astype(int)
        length_rls = int(len(all_run_lengths))
        model_rl_log_distributions = (-np.inf) * np.ones((self.Q, length_rls))
        for index in indices_initialized_models:
            model = self.model_universe[index]
            model_indices_indicators_relative_to_all_run_lengths = np.in1d(all_run_lengths, model.retained_run_lengths)
            if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                model_indices_indicators_relative_to_all_run_lengths[-1] = True
            model_rl_log_distributions[index, model_indices_indicators_relative_to_all_run_lengths] = (
                    model.joint_log_probabilities - self.log_evidence)
        run_length_log_distr = special.logsumexp(model_rl_log_distributions, axis=0)
        self.model_and_run_length_log_distr = model_rl_log_distributions
        self.run_length_log_distr = run_length_log_distr
        self.all_retained_run_lengths = all_run_lengths
        if self.store_mrl or self.store_rl:
            self.storage_all_retained_run_lengths.append(self.all_retained_run_lengths)
        if self.store_rl:
            self.storage_run_length_log_distr.append(self.run_length_log_distr)
        if self.store_mrl:
            self.storage_model_and_run_length_log_distr.append(self.model_and_run_length_log_distr)

    def update_all_joint_log_probabilities(self, y, t):
        index_initialized = []
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            if model.has_lags and ((t + 1) - model.lag_length >= 1):
                index_initialized.append(m)
            elif not model.has_lags:
                index_initialized.append(m)
        prior_rescaling_factor = np.sum(self.model_prior[index_initialized])
        log_model_posteriors_der_m = None
        log_model_posteriors_der_sign_m = None
        log_CP_evidence_der = None
        log_CP_evidence_der_sign = None

        if self.generalized_bayes_rld == "power_divergence" and self.alpha_rld_learning and t > 1:
            at_least_one_model_initialized = np.any([(model.has_lags and
                                                      (t - model.lag_length) > 1) or (not model.has_lags)
                                                     for model in self.model_universe])
            if at_least_one_model_initialized:
                all_log_probs = -np.inf * np.ones((self.Q, np.size(self.all_retained_run_lengths)))
                all_log_alpha_derivatives = -np.inf * np.ones((self.Q, np.size(self.all_retained_run_lengths)))
                all_log_alpha_derivatives_sign = np.zeros((self.Q, np.size(self.all_retained_run_lengths)))
                for m, model in zip(range(0, self.Q), self.model_universe):
                    warmed_up = ((model.has_lags and (t - model.lag_length) > 1) or (not model.has_lags))
                    if warmed_up:
                        model_indices_indicators_relative_to_all_run_lengths = (
                            np.in1d(self.all_retained_run_lengths, model.retained_run_lengths))
                        if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                            model_indices_indicators_relative_to_all_run_lengths[-1] = True
                        if model.log_alpha_derivatives_joint_probabilities is None:
                            num_needed = np.sum(model_indices_indicators_relative_to_all_run_lengths)
                            model.log_alpha_derivatives_joint_probabilities = (-np.inf * np.ones(num_needed))
                            model.log_alpha_derivatives_joint_probabilities_sign = (np.ones(num_needed))
                        all_log_alpha_derivatives[m, model_indices_indicators_relative_to_all_run_lengths] = (
                            model.log_alpha_derivatives_joint_probabilities)
                        all_log_alpha_derivatives_sign[m, model_indices_indicators_relative_to_all_run_lengths] = (
                            model.log_alpha_derivatives_joint_probabilities_sign)
                        all_log_probs[m, model_indices_indicators_relative_to_all_run_lengths] = (
                            model.joint_log_probabilities)
                model_sums_derivatives, model_sums_derivatives_sign = (
                    scipy.special.logsumexp(
                        a=all_log_alpha_derivatives,
                        b=all_log_alpha_derivatives_sign,
                        return_sign=True,
                        axis=0
                    ))
                model_sums = scipy.special.logsumexp(a=all_log_probs, axis=0)
                expr_1 = all_log_alpha_derivatives - model_sums
                sign_1 = all_log_alpha_derivatives_sign
                expr_2 = -2 * model_sums + model_sums_derivatives + all_log_probs
                sign_2 = (-1) * model_sums_derivatives_sign
                expr, sign = scipy.special.logsumexp(
                    a=np.array([expr_1, expr_2]),
                    b=np.array([sign_1, sign_2 * np.ones(self.Q)[:, np.newaxis]]),
                    return_sign=True,
                    axis=0
                )
                log_model_posteriors_der = expr
                log_model_posteriors_der_sign = sign
                _1, _2 = special.logsumexp(
                    a=np.log(self.cp_model.hazard_vector(1, t)) + all_log_alpha_derivatives,
                    b=all_log_alpha_derivatives_sign,
                    return_sign=True)
                log_CP_evidence_der = _1
                log_CP_evidence_der_sign = _2
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            initialization_required = False
            if model.has_lags and (t - model.lag_length == 1):
                initialization_required = True
            elif (not model.has_lags) and t == 1:
                initialization_required = True
            if initialization_required:
                if model.has_lags or isinstance(model, BVARNIG):
                    X_endo = self.data[:model.lag_length + 1, :]
                    Y_2 = self.data[model.lag_length + 1, :]
                    if model.exo_bool:
                        X_exo = self.exo_data[model.lag_length, model.exo_selection, :]
                        X_exo_2 = self.exo_data[model.lag_length + 1, model.exo_selection, :]
                    else:
                        X_exo = X_exo_2 = None
                    model.initialization(X_endo, X_exo, Y_2, X_exo_2, self.cp_model,
                                         self.model_prior[m] / prior_rescaling_factor)
                else:
                    model.initialization(y, self.cp_model, self.model_prior[m])
            else:
                warmed_up = ((model.has_lags and (t - model.lag_length) > 1) or (not model.has_lags))
                if warmed_up:
                    log_model_posteriors = (self.model_and_run_length_log_distr[m, :] - self.run_length_log_distr)
                    log_model_posteriors = log_model_posteriors[np.where(log_model_posteriors > -np.inf)]
                    log_CP_evidence = special.logsumexp(np.log(
                        self.cp_model.hazard_vector(1, t)) + self.model_and_run_length_log_distr + self.log_evidence)
                    if self.generalized_bayes_rld == "power_divergence" and self.alpha_rld_learning:
                        model_indices_indicators_relative_to_all_run_lengths = (
                            np.in1d(self.all_retained_run_lengths, model.retained_run_lengths))
                        if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                            model_indices_indicators_relative_to_all_run_lengths[-1] = True
                        log_model_posteriors_der_m = (
                            log_model_posteriors_der[m, model_indices_indicators_relative_to_all_run_lengths])
                        log_model_posteriors_der_sign_m = (
                            log_model_posteriors_der_sign[m, model_indices_indicators_relative_to_all_run_lengths])
                if model.has_lags and ((t) - model.lag_length) > 1:
                    if (t - model.lag_length) > 2 and self.alpha_param_opt_t <= t:
                        model.alpha_param_gradient_computation(y=y, t=t,
                                                               cp_model=self.cp_model,
                                                               model_prior=
                                                               self.model_prior[m] / prior_rescaling_factor,
                                                               log_model_posteriors=log_model_posteriors,
                                                               log_CP_evidence=log_CP_evidence,
                                                               eps=self.eps_param_learning(t))
                    model.update_joint_log_probabilities(
                        y=y, t=t, cp_model=self.cp_model,
                        model_prior=
                        self.model_prior[m] / prior_rescaling_factor,
                        log_model_posteriors=log_model_posteriors,
                        log_CP_evidence=log_CP_evidence,
                        log_model_posteriors_der=log_model_posteriors_der_m,
                        log_model_posteriors_der_sign=
                        log_model_posteriors_der_sign_m,
                        log_CP_evidence_der=log_CP_evidence_der,
                        log_CP_evidence_der_sign=log_CP_evidence_der_sign,
                        do_general_bayesian_hyperparameter_optimization=(
                            warmed_up))
                elif not model.has_lags:
                    if t > 2 and self.alpha_param_opt_t <= t:
                        model.alpha_param_gradient_computation(y=y, t=t,
                                                               cp_model=self.cp_model,
                                                               model_prior=
                                                               self.model_prior[m] / prior_rescaling_factor,
                                                               log_model_posteriors=log_model_posteriors,
                                                               log_CP_evidence=log_CP_evidence,
                                                               eps=self.eps_param_learning(t))
                    model.update_joint_log_probabilities(
                        y=y, t=t, cp_model=self.cp_model,
                        model_prior=
                        self.model_prior[m] / prior_rescaling_factor,
                        log_model_posteriors=log_model_posteriors,
                        log_CP_evidence=log_CP_evidence,
                        log_model_posteriors_der=log_model_posteriors_der_m,
                        log_model_posteriors_der_sign=
                        log_model_posteriors_der_sign_m,
                        log_CP_evidence_der=log_CP_evidence_der,
                        log_CP_evidence_der_sign=log_CP_evidence_der_sign,
                        do_general_bayesian_hyperparameter_optimization=(
                            warmed_up))
                if model.has_lags and (t - model.lag_length) > 1:
                    y_tm1 = self.data[t - 2, :]
                    if model.exo_bool:
                        x_exo_t = self.exo_data[t, model.exo_selection, :]
                        x_exo_tp1 = self.exo_data[t + 1, model.exo_selection, :]
                    else:
                        x_exo_t = x_exo_tp1 = None
                    if isinstance(model, BVARNIGDPD):
                        model.update_predictive_distributions(y, y_tm1, x_exo_t, x_exo_tp1, t, self.cp_model.hazard(0))
                    else:
                        model.update_predictive_distributions(y, y_tm1, x_exo_t, x_exo_tp1, t)
                elif not model.has_lags:
                    if isinstance(model, BVARNIG):
                        y_tm1 = self.data[t - 2, :]
                        if model.exo_bool:
                            x_exo_t = self.exo_data[t, model.exo_selection, :]
                            x_exo_tp1 = self.exo_data[t + 1, model.exo_selection, :]
                        else:
                            x_exo_t = x_exo_tp1 = None
                        if isinstance(model, BVARNIGDPD):
                            model.update_predictive_distributions(y, y_tm1, x_exo_t, x_exo_tp1, t,
                                                                  self.cp_model.hazard(0))
                        else:
                            model.update_predictive_distributions(y, y_tm1, x_exo_t, x_exo_tp1, t)
                    else:
                        model.update_predictive_distributions(y, t)

    def update_alpha_rld(self, y, t, update=True):
        num_run_lengths = int(len(self.all_retained_run_lengths))
        all_log_alpha_derivatives = -np.inf * np.ones((self.Q, num_run_lengths))
        all_log_alpha_derivatives_sign = np.zeros((self.Q, num_run_lengths))
        all_log_probs = -np.inf * np.ones((self.Q, num_run_lengths))

        for m, model in zip(range(0, self.Q), self.model_universe):
            model_indices_indicators_relative_to_all_run_lengths = np.in1d(self.all_retained_run_lengths,
                                                                           model.retained_run_lengths)
            if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                model_indices_indicators_relative_to_all_run_lengths[-1] = True
            all_log_alpha_derivatives[m, model_indices_indicators_relative_to_all_run_lengths] = (
                model.log_alpha_derivatives_joint_probabilities)
            all_log_alpha_derivatives_sign[m, model_indices_indicators_relative_to_all_run_lengths] = (
                model.log_alpha_derivatives_joint_probabilities_sign)
            all_log_probs[m, model_indices_indicators_relative_to_all_run_lengths] = (
                model.joint_log_probabilities)
        sum_derivatives, sum_derivatives_sign = scipy.special.logsumexp(
            a=all_log_alpha_derivatives,
            b=all_log_alpha_derivatives_sign,
            return_sign=True)
        term_1 = all_log_alpha_derivatives - self.log_evidence
        term_1_sign = all_log_alpha_derivatives_sign
        term_2 = sum_derivatives - 2.0 * self.log_evidence + all_log_probs
        term_2_sign = (-1) * sum_derivatives_sign * np.abs(term_1_sign)
        run_length_and_model_log_der, run_length_and_model_log_der_sign = (
            scipy.special.logsumexp(
                a=np.array([term_1, np.abs(term_1_sign) * term_2]),
                b=np.array([term_1_sign, term_2_sign]),
                return_sign=True,
                axis=0
            ))
        resid = self.y_pred_mean.flatten() - y.flatten()
        post_mean_der = np.zeros(shape=(self.S1 * self.S2))
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            num_rl = np.size(model.retained_run_lengths)
            model_indices_indicators_relative_to_all_run_lengths = np.in1d(self.all_retained_run_lengths,
                                                                           model.retained_run_lengths)
            if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                model_indices_indicators_relative_to_all_run_lengths[-1] = True
            post_mean_der = (post_mean_der + np.sum(
                np.reshape(model.get_posterior_expectation(t), newshape=(num_rl, self.S1 * self.S2)) * (np.exp(
                    run_length_and_model_log_der[m, model_indices_indicators_relative_to_all_run_lengths]) *
                                                                                                        run_length_and_model_log_der_sign[
                                                                                                            m, model_indices_indicators_relative_to_all_run_lengths])[
                                                                                                       :, np.newaxis],
                axis=0))
        self.gradient_alpha_rld = (
                    self.gradient_alpha_rld + self.loss_der_rld_learning(resid.flatten(), post_mean_der.flatten(),
                                                                         self.C))
        self.gradient_alpha_rld_count = self.gradient_alpha_rld_count + 1
        step_size = self.step_size_gen(t=t)
        min_increment, max_increment = 0.0000, 5 / self.T
        min_alpha, max_alpha = pow(10, -5), 5
        if update:
            grad_sign = np.sign(self.gradient_alpha_rld)
            increment = max(min(max_increment,
                                step_size *
                                np.abs(self.gradient_alpha_rld) *
                                (1.0 / self.gradient_alpha_rld_count)),
                            min_increment)
            self.alpha_rld = min(max(self.alpha_rld - increment * grad_sign, min_alpha), max_alpha)
            self.alpha_list.append(self.alpha_rld)
            for model in self.model_universe:
                model.alpha_rld = self.alpha_rld
            self.gradient_alpha_rld_count = 0
            self.gradient_alpha_rld = 0

    @staticmethod
    def step_size_gen(t, alpha=None):
        if alpha is not None:
            g0 = min(alpha, 1.0)
            lamb = 10
            step_size = g0 / (1.0 + g0 * lamb * t)
            return step_size
        else:
            g0 = 3.0
            lamb = 0.5
            step_size = g0 / (1.0 + g0 * lamb * t)
            return step_size

    @staticmethod
    def step_size_gen_rld(t, alpha=None):
        g0 = 0.05
        lamb = 0.5
        step_size = g0 / (1.0 + g0 * lamb * t)
        return step_size

    @staticmethod
    def eps_gen(t):
        return pow(t, -0.25)

    @staticmethod
    def squared_loss(resid, C):
        return 0.5 * np.sum(np.power(resid, 2))

    @staticmethod
    def absolute_loss(resid, C):
        return np.sum(np.abs(resid))

    @staticmethod
    def biweight_loss(resid, C):
        smallerC = np.where(resid < C)
        biggerC = int(len(resid) - len(smallerC))
        return 0.5 * np.sum(np.power(resid[smallerC], 2)) + 0.5 * biggerC * pow(C, 2)

    @staticmethod
    def bounded_absolute_loss(resid, C):
        smallerC = np.where(resid < C)
        biggerC = int(len(resid) - len(smallerC))
        return np.sum(np.abs(resid[smallerC])) + biggerC * C

    @staticmethod
    def squared_loss_derivative(resid, post_mean_der, C):
        return np.sum(2 * resid * post_mean_der)

    @staticmethod
    def absolute_loss_derivative(resid, post_mean_der, C):
        return np.sum(np.sign(resid) * post_mean_der)

    @staticmethod
    def biweight_loss_derivative(resid, post_mean_der, C):
        smallerC = np.where(resid < C)
        return np.sum(2 * resid[smallerC] * post_mean_der[smallerC])

    @staticmethod
    def bounded_absolute_loss_derivative(resid, post_mean_der, C):
        smallerC = np.where(resid < C)
        return np.sum(np.abs(resid[smallerC] * post_mean_der[smallerC]))

    def save_negative_log_likelihood(self, t):
        all_one_step_ahead_lklhoods_weighted = [(self.model_universe[m].one_step_ahead_predictive_log_probs +
                                                 self.model_and_run_length_log_distr[
                                                     m, self.model_and_run_length_log_distr[m, :] > -np.inf]) for m in
                                                range(0, self.Q)]
        summed_up = -special.logsumexp([item for entry in all_one_step_ahead_lklhoods_weighted for item in entry])
        self.negative_log_likelihood.append(summed_up)

    def compute_negative_log_likelihood_fixed_pars(self, y, t):
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            if model.has_lags:
                model.save_NLL_fixed_pars(y, t)

    def save_negative_log_likelihood_fixed_pars(self, t):
        all_one_step_ahead_lklhoods_weighted = [(self.model_universe[m].one_step_ahead_predictive_log_probs_fixed_pars +
                                                 self.model_and_run_length_log_distr[
                                                     m, self.model_and_run_length_log_distr[m, :] > -np.inf]) for m in
                                                range(0, self.Q)]
        summed_up = -special.logsumexp([item for entry in all_one_step_ahead_lklhoods_weighted for item in entry])
        self.negative_log_likelihood_fixed_pars.append(summed_up)

    def save_MSE(self, y, t):
        self.MSE.append(pow(self.y_pred_mean - y.reshape(self.S1, self.S2), 2))
        self.MAE.append(abs(self.y_pred_mean - y.reshape(self.S1, self.S2)))

    def update_log_evidence(self):
        self.log_evidence = scipy.special.logsumexp([model.model_log_evidence for model in self.model_universe])

    def update_model_and_run_length_log_distribution(self, t):
        r_max, r_max2 = 0, 0
        for q in range(0, self.Q):
            retained_q = self.model_universe[q].retained_run_lengths
            r_max2 = max(np.max(retained_q), r_max)
            if r_max2 >= r_max:
                r_max = r_max2
                if (retained_q.shape[0] > 1) and (retained_q[-1] == retained_q[-2]):
                    r_max = r_max + 1
        self.model_and_run_length_log_distr = (-np.inf * np.ones(shape=(self.Q, r_max + 1)))
        for i in range(0, self.Q):
            model = self.model_universe[i]
            retained = model.retained_run_lengths
            if (retained.shape[0] > 1) and (retained[-1] == retained[-2]):
                retained = np.copy(model.retained_run_lengths)
                retained[-1] = retained[-1] + 1
            self.model_and_run_length_log_distr[i, retained] = model.joint_log_probabilities - model.model_log_evidence
        if self.store_mrl:
            print("storage for model-and-run-length log distr not implemented")

    def prediction_y(self, y, t):
        post_mean, post_var = (
        np.zeros(shape=(self.S1, self.S2)), np.zeros(shape=(self.S1 * self.S2, self.S1 * self.S2)))
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            num_rl = np.size(model.retained_run_lengths)
            post_mean = (post_mean +
                         np.sum(
                             np.reshape(model.get_posterior_expectation(t),
                                        newshape=(num_rl, self.S1, self.S2)) *
                             np.exp(model.joint_log_probabilities -
                                    self.log_evidence)[:, np.newaxis, np.newaxis],
                             axis=0))
            post_var = (post_var +
                        np.sum(
                            np.reshape(model.get_posterior_variance(t),
                                       newshape=(num_rl, self.S1 * self.S2,
                                                 self.S1 * self.S2)) *
                            np.exp(model.joint_log_probabilities -
                                   self.log_evidence)[:, np.newaxis, np.newaxis],
                            axis=0))
        self.y_pred_mean, self.y_pred_var = post_mean, post_var

    def storage(self, t):
        self.storage_mean[t - 1, :, :] = self.y_pred_mean
        self.storage_var[t - 1, :] = np.diag(self.y_pred_var)
        self.storage_log_evidence[t - 1] = self.log_evidence

    def trim_run_length_log_distributions(self, t):
        for model in self.model_universe:
            if model.has_lags:
                if model.lag_length < t:
                    model.trim_run_length_log_distrbution(t, self.threshold, self.trim_type)
            else:
                model.trim_run_length_log_distrbution(t, self.threshold, self.trim_type)

    def update_priors(self, t):
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            if model.auto_prior_update:
                model_specific_rld = (self.model_and_run_length_log_distr[m, :] - special.logsumexp(
                    self.model_and_run_length_log_distr[m, :]))
                model.prior_update(t, model_specific_rld)

    def MAP_estimate(self, t):
        log_MAP_current, log_MAP_proposed = -np.inf, -np.inf
        initializer, CP_initialization = False, False
        CP_proposed = [-99, -99]
        r_cand, r_star = -99, -99
        m, m_star = 0, self.m_star_old
        all_rl = np.array([])
        for model in self.model_universe:
            all_rl = np.union1d(all_rl, model.retained_run_lengths)
            if model.retained_run_lengths[-1] == model.retained_run_lengths[-2]:
                r_more = np.array([t])
                all_rl = np.union1d(all_rl, r_more)
        all_rl = all_rl.astype(int)
        for (m, model) in zip(range(0, self.Q), self.model_universe):
            if model.has_lags:
                model_lag = model.lag_length
            else:
                model_lag = 0
            if model_lag + 1 == t:
                initializer = True
                log_MAP_0 = model.joint_log_probabilities[0]
                log_MAP_larger_0 = model.joint_log_probabilities[1]
                if log_MAP_0 > log_MAP_larger_0:
                    r_cand = 0
                    log_MAP_proposed = log_MAP_0
                else:
                    r_cand = t + 1
                    log_MAP_proposed = log_MAP_larger_0
            if model_lag + 1 < t:
                initializer = False
                log_densities = -np.inf * np.ones(np.size(all_rl))
                model_run_lengths = model.retained_run_lengths.copy()
                if model_run_lengths[-1] == model_run_lengths[-2]:
                    model_run_lengths[-1] = t
                index_indicators = np.in1d(all_rl, model_run_lengths)
                log_densities[index_indicators] = model.joint_log_probabilities - self.log_evidence
                MAP_factor = np.flipud(self.log_MAP_storage)[all_rl]
                candidates = (log_densities + MAP_factor)
                log_MAP_proposed = np.max(candidates)
                r_cand = (all_rl)[np.argmax(candidates)]
                if r_cand == t - 1:
                    r_cand = t - 2
                if r_cand == t:
                    r_cand = t - 1
            if log_MAP_proposed > log_MAP_current:
                log_MAP_current = log_MAP_proposed
                CP_initialization = initializer
                m_star = m
                r_star = r_cand
                CP_proposed = [t - r_star, m_star]
            m += 1
        self.log_MAP_storage = np.append(self.log_MAP_storage, log_MAP_current)
        if CP_initialization:
            self.CPs[t - 1] = [CP_proposed]
        elif log_MAP_current > -np.inf:
            if r_star == self.r_star_old + 1 and m_star == self.m_star_old:
                self.CPs[t - 1] = self.CPs[t - 2]
            else:
                self.CPs[t - 1] = self.CPs[t - 2 - r_star].copy() + [CP_proposed]
        self.r_star_old = r_star
        self.m_star_old = m_star

    @staticmethod
    def default_step_size_param_learning(t, alpha=None):
        if alpha is None:
            return pow(t, -1)
        else:
            return alpha * pow(t, -1)

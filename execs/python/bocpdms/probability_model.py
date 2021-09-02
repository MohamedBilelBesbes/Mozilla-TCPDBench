import numpy as np
from scipy import special


class ProbabilityModel:
    def evaluate_predictive_log_distribution(self, y, r):
        pass

    def initialization(self, y, cp_model, model_prior):
        pass

    def rescale_DPD_run_length_distribution(self, log_scale_new, rescaler_for_old_obs, t):
        if rescaler_for_old_obs is None:
            self.joint_log_probabilities = self.joint_log_probabilities - log_scale_new
        else:
            self.joint_log_probabilities[0] = self.joint_log_probabilities[0] - log_scale_new
            self.joint_log_probabilities[1:] = self.joint_log_probabilities[1:] + rescaler_for_old_obs
        self.model_log_evidence = special.logsumexp(self.joint_log_probabilities)

    def update_joint_log_probabilities(
            self,
            y, t, cp_model, model_prior,
            log_model_posteriors,
            log_CP_evidence,
            log_model_posteriors_der=None,
            log_model_posteriors_der_sign=None,
            log_CP_evidence_der=None,
            log_CP_evidence_der_sign=None,
            do_general_bayesian_hyperparameter_optimization=False,
            disable_hyperparameter_optimization=False):
        self.r0_log_prob = self.evaluate_log_prior_predictive(y, t)
        self.one_step_ahead_predictive_log_probs = self.evaluate_predictive_log_distribution(y, t)
        if self.generalized_bayes_rld == "power_divergence":
            integrals = self.get_log_integrals_power_divergence()
            max_val = np.log(np.finfo(np.float64).max)
            integral_exceeds = np.where(integrals >= max_val)[0]
            integral_fine = np.where(integrals < max_val)[0]
            integral_exp = integrals
            if len(integral_exceeds) > 0:
                integral_exp[integral_exceeds] = min(1.0, (1.0 / (self.alpha_rld + 1.0))) * np.finfo(np.float64).max
            if len(integral_fine) > 0:
                integral_exp[integral_fine] = (1.0 / (self.alpha_rld + 1.0)) * np.exp(integrals[integral_fine])
            if self.r0_log_prob >= max_val:
                r0_log_prob_exp = min(1.0, 1.0 / self.alpha_rld) * (min(self.alpha_rld, 1) * max_val)
            else:
                r0_log_prob_exp = (1.0 / self.alpha_rld) * np.exp(self.r0_log_prob * self.alpha_rld)
            step_ahead_exceeds = np.where(self.one_step_ahead_predictive_log_probs >= max_val)[0]
            step_ahead_fine = np.where(self.one_step_ahead_predictive_log_probs < max_val)[0]
            step_ahead_exp = self.one_step_ahead_predictive_log_probs.copy()
            if len(step_ahead_exceeds) > 0:
                if self.alpha_rld < 1:
                    step_ahead_exp[step_ahead_exceeds] = min(1.0, 1.0 / (self.alpha_rld)) * (np.finfo(np.float64).max * self.alpha_rld)
                else:
                    step_ahead_exp[step_ahead_exceeds] = min(1.0, (1.0 / (self.alpha_rld))) * np.finfo(np.float64).max
            if len(step_ahead_fine) > 0:
                step_ahead_exp[step_ahead_fine] = (
                    ((1.0 / (self.alpha_rld)) *
                     np.exp(
                         self.one_step_ahead_predictive_log_probs[
                             step_ahead_fine] *
                         self.alpha_rld)
                     ))
            self.one_step_ahead_predictive_log_loss = step_ahead_exp - integral_exp[1:]
            self.r0_log_loss = r0_log_prob_exp - integral_exp[0]
            if (self.alpha_rld_learning and
                    do_general_bayesian_hyperparameter_optimization and
                    disable_hyperparameter_optimization is not True):
                self.update_alpha_derivatives(y, t,
                                              log_model_posteriors,
                                              log_model_posteriors_der,
                                              log_model_posteriors_der_sign,
                                              log_CP_evidence,
                                              log_CP_evidence_der,
                                              log_CP_evidence_der_sign,
                                              model_prior,
                                              cp_model)
        elif self.generalized_bayes_rld == "kullback_leibler":
            self.r0_log_loss = self.r0_log_prob
            self.one_step_ahead_predictive_log_loss = self.one_step_ahead_predictive_log_probs.copy()
        try:
            growth_log_probabilities = (self.one_step_ahead_predictive_log_loss +
                                        self.joint_log_probabilities +
                                        log_model_posteriors +
                                        np.log(1 - cp_model.hazard_vector(1, t)))
        except ValueError as v:
            print(v)
            print("log model posteriors:", log_model_posteriors)
            print("log model posteriors shape:", log_model_posteriors.shape)
        CP_log_prob = self.r0_log_loss + np.log(model_prior) + log_CP_evidence
        joint_log_probabilities_tm1 = self.joint_log_probabilities
        self.joint_log_probabilities = np.insert(growth_log_probabilities, 0, CP_log_prob)
        model_log_evidence_tm1 = self.model_log_evidence
        self.model_log_evidence = special.logsumexp(self.joint_log_probabilities)
        if (self.hyperparameter_optimization is not None and
                self.hyperparameter_optimization is not False and
                disable_hyperparameter_optimization is not True):
            run_length_log_distro = joint_log_probabilities_tm1 - model_log_evidence_tm1
            gradients_log_predictive_val, gradients_log_predictive_sign = (
                self.differentiate_predictive_log_distribution(y, t, run_length_log_distro))
            num_params = gradients_log_predictive_val.shape[0]
            run_length_num = self.retained_run_lengths.shape[0]
            model_specific_joint_log_probs = joint_log_probabilities_tm1 - np.log(model_prior)
            jlpd_part1_val = gradients_log_predictive_val[:, 1:] + model_specific_joint_log_probs + np.log(1 - cp_model.hazard_vector(1, t))
            jlpd_part1_sign = gradients_log_predictive_sign[:, 1:]
            jlpd_part2_val = self.model_specific_joint_log_probabilities_derivative + self.r0_log_prob + np.log(1 - cp_model.hazard_vector(1, t))
            jlpd_part2_sign = self.model_specific_joint_log_probabilities_derivative_sign
            res_val, res_sign = special.logsumexp(
                a=np.array([jlpd_part1_val, jlpd_part2_val]),
                b=np.array([jlpd_part1_sign, jlpd_part2_sign]),
                return_sign=True,
                axis=0
            )
            results_grad_1 = (
                special.logsumexp(
                    a=np.array([
                        gradients_log_predictive_val[:, 1:] +
                        np.log(cp_model.hazard_vector(1, t)) +
                        model_specific_joint_log_probs
                    ]),
                    b=gradients_log_predictive_sign[:, 1:],
                    return_sign=True,
                    axis=1
                )
            )
            CP_grad_1_val, CP_grad_1_sign = results_grad_1[0].flatten(), results_grad_1[1].flatten()
            results_grad_2 = (
                special.logsumexp(
                    a=np.array([
                        self.model_specific_joint_log_probabilities_derivative +
                        np.log(cp_model.hazard_vector(1, t)) +
                        self.one_step_ahead_predictive_log_probs
                    ]),
                    b=(
                        self.model_specific_joint_log_probabilities_derivative_sign),
                    return_sign=True,
                    axis=1
                )
            )
            CP_grad_2_val, CP_grad_2_sign = results_grad_2[0].flatten(), results_grad_2[1].flatten()

            CP_grad_val, CP_grad_sign = special.logsumexp(
                a=np.array([CP_grad_1_val, CP_grad_2_val]),
                b=np.array([CP_grad_1_sign, CP_grad_2_sign]),
                return_sign=True,
                axis=1
            )
            if self.hyperparameter_optimization == "caron" or self.hyperparameter_optimization == "online":
                joint_log_der_sum, joint_log_der_sum_signs = (
                    special.logsumexp(
                        a=self.model_specific_joint_log_probabilities_derivative,
                        b=self.model_specific_joint_log_probabilities_derivative_sign,
                        return_sign=True, axis=1))
                log_evidence = special.logsumexp(model_specific_joint_log_probs)
                part1 = self.model_specific_joint_log_probabilities_derivative - log_evidence
                part2 = joint_log_probabilities_tm1 - 2 * log_evidence + joint_log_der_sum[:, np.newaxis]
                rld_der_signs, rld_der = special.logsumexp(
                    a=np.array([part1, part2]),
                    b=np.array([
                        self.model_specific_joint_log_probabilities_derivative_sign,
                        (-1) * joint_log_der_sum_signs[:, np.newaxis] *
                        np.ones((num_params, run_length_num))]),
                    return_sign=True, axis=0)
                gradient, gradient_signs = special.logsumexp(
                    a=np.array([
                        gradients_log_predictive_val[:, 1:] +
                        run_length_log_distro[np.newaxis, :],
                        self.one_step_ahead_predictive_log_probs[np.newaxis, :] +
                        rld_der]),
                    b=np.array([
                        gradients_log_predictive_sign[:, 1:],
                        rld_der_signs]),
                    return_sign=True,
                    axis=(0, 2))
                pred = special.logsumexp(self.one_step_ahead_predictive_log_probs + run_length_log_distro)
                grad = np.nan_to_num(np.exp(gradient - pred))
                sig = np.nan_to_num(gradient_signs)
                grad[grad == 0] = pow(10, 5)
                caron_gradient = np.minimum((grad), pow(10, 5) * np.ones(num_params)) * sig
                p, C, scale = 1.0005, 1000, 3
                step_size = self.step_caron_pow2(t, p, C, scale)
                self.caron_hyperparameter_optimization(t, caron_gradient, step_size)
            self.model_specific_joint_log_probabilities_derivative = np.insert(res_val, 0, CP_grad_val, axis=1)
            self.model_specific_joint_log_probabilities_derivative_sign = np.insert(res_sign, 0, CP_grad_sign, axis=1)

    def update_alpha_derivatives(self, y, t,
                                 log_model_posteriors,
                                 log_model_posteriors_der,
                                 log_model_posteriors_der_sign,
                                 log_CP_evidence,
                                 log_CP_evidence_der,
                                 log_CP_evidence_der_sign,
                                 model_prior,
                                 cp_model):
        _1, _2 = self.get_one_step_ahead_log_loss_derivatives_power_divergence()
        one_step_ahead_log_loss_derivatives = _1
        one_step_ahead_log_loss_derivatives_sign = _2
        full = self.one_step_ahead_predictive_log_loss + self.joint_log_probabilities + log_model_posteriors + np.log(1 - cp_model.hazard_vector(1, t))
        rest_1 = full - self.joint_log_probabilities
        rest_2 = full - self.one_step_ahead_predictive_log_loss
        rest_3 = full - log_model_posteriors
        new_derivatives, new_derivatives_sign = special.logsumexp(
            a=np.array([
                rest_1 + self.log_alpha_derivatives_joint_probabilities,
                rest_2 + one_step_ahead_log_loss_derivatives[1:],
                rest_3 + log_model_posteriors_der
            ]),
            b=np.array([
                self.log_alpha_derivatives_joint_probabilities_sign,
                one_step_ahead_log_loss_derivatives_sign[1:],
                log_model_posteriors_der_sign
            ]),
            return_sign=True,
            axis=0
        )
        CP_d_1 = (one_step_ahead_log_loss_derivatives[0] + np.log(model_prior) + log_CP_evidence)
        CP_d_2 = (self.r0_log_loss + np.log(model_prior) + log_CP_evidence_der)
        CP_derivative, CP_derivative_sign = special.logsumexp(
            a=np.array([CP_d_1, CP_d_2]),
            b=np.array([one_step_ahead_log_loss_derivatives_sign[0],
                        log_CP_evidence_der_sign]),
            return_sign=True)
        self.log_alpha_derivatives_joint_probabilities = np.insert(new_derivatives, 0, CP_derivative)
        self.log_alpha_derivatives_joint_probabilities_sign = np.insert(new_derivatives_sign, 0, CP_derivative_sign)

    def step_caron_pow2(self, t, p, C, scale):
        if self.has_lags:
            t_ = t - self.lag_length
        else:
            t_ = t
        return scale * C * (1 / (t_ + C))

    def update_predictive_distributions(self, y, t, r_evaluations):
        pass

    def get_posterior_expectation(self, t, r_list=None):
        pass

    def get_posterior_variance(self, t, r_list=None):
        pass

    def prior_log_density(self, y_flat):
        pass

    def trim_run_length_log_distrbution(self, t, threshold, trim_type):
        if not ((threshold is None) or (threshold == 0) or (threshold == -1)):
            run_length_log_distr = self.joint_log_probabilities - self.model_log_evidence
            if trim_type == "threshold":
                kept_run_lengths = np.array(run_length_log_distr) >= threshold
            elif trim_type == "keep_K":
                K = min(threshold, np.size(run_length_log_distr))
                max_indices = (-run_length_log_distr).argsort()[:K]
                kept_run_lengths = np.full(np.size(run_length_log_distr), False, dtype=bool)
                kept_run_lengths[max_indices] = True
            elif trim_type == "SRC":
                if threshold < 0:
                    alpha = threshold
                elif threshold > 0:
                    K = min(threshold, np.size(run_length_log_distr))
                    alpha_index = (-run_length_log_distr).argsort()[K - 1]
                    alpha = run_length_log_distr[alpha_index]
                kept_run_lengths = np.array(run_length_log_distr) >= alpha
                u = np.random.uniform(low=0, high=alpha)
                for (index, rl_particle) in zip(range(0, np.size(
                        run_length_log_distr)), run_length_log_distr):
                    if rl_particle < alpha:
                        if u <= rl_particle:
                            kept_run_lengths[index] = True
                            run_length_log_distr[index] = alpha
                            u = special.logsumexp(np.array([u, alpha, -rl_particle]))
            if trim_type == "SRC" and np.sum(kept_run_lengths) > 1:
                self.joint_log_probabilities = run_length_log_distr + self.model_log_evidence
                self.trimmer(kept_run_lengths)
            elif np.sum(kept_run_lengths) > 1:
                self.trimmer(kept_run_lengths)
            elif np.sum(kept_run_lengths) <= 1:
                max_indices = (-run_length_log_distr).argsort()[:2]
                kept_run_lengths[max_indices] = True
                self.trimmer(kept_run_lengths)

    def alpha_param_gradient_computation(self, y, t, cp_model, model_prior, log_model_posteriors, log_CP_evidence, eps):
        pass

    def DPD_joint_log_prob_updater(self, alpha_direction, y, t, cp_model, model_prior, log_model_posteriors, log_CP_evidence):
        r0_log_prob = self.evaluate_log_prior_predictive(y, t, store_posterior_predictive_quantities=False)
        one_step_ahead_predictive_log_probs = self.evaluate_predictive_log_distribution(y, t, store_posterior_predictive_quantities=False, alpha_direction=alpha_direction)
        if self.generalized_bayes_rld == "power_divergence":
            integrals = self.get_log_integrals_power_divergence(DPD_call=True)
            max_val = np.log(np.finfo(np.float64).max)
            integral_exceeds = np.where(integrals >= max_val)[0]
            integral_fine = np.where(integrals < max_val)[0]
            integral_exp = integrals
            if len(integral_exceeds) > 0:
                integral_exp[integral_exceeds] = min(1.0, (1.0 / (self.alpha_rld + 1))) * np.finfo(np.float64).max
            if len(integral_fine) > 0:
                integral_exp[integral_fine] = min(1.0, (1.0 / (self.alpha_rld + 1))) * np.exp(integrals[integral_fine])
            if self.r0_log_prob >= max_val:
                r0_log_prob_exp = min(1.0, 1.0 / self.alpha_rld) * (min(self.alpha_rld, 1) * max_val)
            else:
                r0_log_prob_exp = (1.0 / self.alpha_rld * np.exp(
                    r0_log_prob * self.alpha_rld))
            step_ahead_exceeds = np.where(one_step_ahead_predictive_log_probs >= max_val)[0]
            step_ahead_fine = np.where(one_step_ahead_predictive_log_probs < max_val)[0]
            step_ahead_exp = one_step_ahead_predictive_log_probs
            if len(step_ahead_exceeds) > 0:
                if self.alpha_rld < 1:
                    step_ahead_exp[step_ahead_exceeds] = min(1.0, 1.0 / self.alpha_rld) * (np.finfo(np.float64).max * self.alpha_rld)
                else:
                    step_ahead_exp[step_ahead_exceeds] = min(1.0, (1.0 / self.alpha_rld)) * np.finfo(np.float64).max
            if len(step_ahead_fine) > 0:
                step_ahead_exp[step_ahead_fine] = ((1.0 / (self.alpha_rld + 1)) * np.exp(one_step_ahead_predictive_log_probs[step_ahead_fine] * self.alpha_rld))
            one_step_ahead_predictive_log_probs = step_ahead_exp - integral_exp[1:]
            r0_log_prob = r0_log_prob_exp - integral_exp[0]
        growth_log_probabilities = (one_step_ahead_predictive_log_probs +
                                    self.joint_log_probabilities +
                                    log_model_posteriors +
                                    np.log(1 - cp_model.hazard_vector(1, t)))
        CP_log_prob = r0_log_prob + np.log(model_prior) + log_CP_evidence
        joint_log_probabilities = np.insert(growth_log_probabilities, 0, CP_log_prob)
        return joint_log_probabilities

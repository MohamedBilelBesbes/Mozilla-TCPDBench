import numpy as np
from scipy import special
from scipy import linalg
from scipy import stats
import scipy
from .probability_model import ProbabilityModel
from .nearestPD import NPD


class BVARNIG(ProbabilityModel):
    def __init__(self, prior_a, prior_b, S1, S2, prior_mean_beta=None, prior_var_beta=None, prior_mean_scale=0,
                 prior_var_scale=100, nbh_sequence=None, restriction_sequence=None, intercept_grouping=None,
                 general_nbh_sequence=None, general_nbh_restriction_sequence=None, exo_selection=None,
                 padding='overall_mean', auto_prior_update=False, hyperparameter_optimization="online",
                 general_nbh_coupling="strong coupling", non_spd_alerts=False):
        self.a, self.b = prior_a, prior_b
        if not prior_mean_beta is None:
            self.prior_mean_beta = prior_mean_beta.flatten()
        else:
            self.prior_mean_beta = prior_mean_beta
        self.prior_var_beta = prior_var_beta
        self.auto_prior_update = auto_prior_update
        if hyperparameter_optimization is not None or hyperparameter_optimization is not False:
            self.a_old = prior_a + 0.0000001
            self.b_old = prior_b + 0.0000001
            self.gradient_old = 0.0
            self.a_list, self.b_list = [], []
        self.hyperparameter_optimization = hyperparameter_optimization
        self.non_spd_alerts = non_spd_alerts
        self.has_lags = True
        self.generalized_bayes_rld = "kullback_leibler"
        self.alpha_rld_learning = False
        self.alpha_rld = None
        self.S1, self.S2 = S1, S2
        self.restriction_sequence = restriction_sequence
        self.nbh_sequence = nbh_sequence
        self.padding = padding
        self.general_nbh_sequence = general_nbh_sequence
        self.general_nbh_restriction_sequence = general_nbh_restriction_sequence
        self.general_nbh_coupling = general_nbh_coupling
        self.intercept_grouping = intercept_grouping
        if (self.restriction_sequence is not None) and (self.nbh_sequence is not None) and (self.padding is not None):
            self.regular_grid = True
        elif (self.general_nbh_sequence is not None) and (self.general_nbh_restriction_sequence is not None) and (
                self.general_nbh_coupling is not None):
            self.regular_grid = False
        elif (self.restriction_sequence is None) and (self.nbh_sequence is None) and (
                self.general_nbh_sequence is None) and (self.general_nbh_restriction_sequence is None):
            self.regular_grid = False
            self.has_lags = False
            self.lag_length = 0
            self.general_nbh_coupling = None
        else:
            raise SystemExit("Your neighbourhood specifications " +
                             "are incomplete: At least one of " +
                             "restriction_sequence, nbh_sequence, padding is None; " +
                             "or at least one of " +
                             "general_nbh_sequence, general_nbh_restriction_sequence ," +
                             " general_nbh_coupling is None")
        if exo_selection is None or exo_selection == []:
            self.exo_bool = False
            exo_selection = []
            self.exo_selection = []
        else:
            self.exo_bool = True
            self.exo_selection = exo_selection
        self.get_intercept_codes()
        self.get_endo_vars()
        self.exo_vars = [self.intercept_codes + exo_selection]
        self.all_vars = list(self.exo_vars) + list(self.endo_vars)
        self.all_vars = sum(self.all_vars, [])
        self.num_exo_regressors = len(sum(self.exo_vars, []))
        self.num_endo_regressors = len(sum(self.endo_vars, []))
        self.num_regressors = self.num_endo_regressors + self.num_exo_regressors
        self.get_lag_counts()
        endo_regressors_per_lag = self.get_endo_regressors_per_lag()
        self.get_extraction_list(endo_regressors_per_lag)
        self.XX, self.YX, self.model_log_evidence = None, None, -np.inf
        self.retained_run_lengths = np.array([0, 0])
        self.joint_log_probabilities = 1
        self.log_alpha_derivatives_joint_probabilities = None
        self.log_alpha_derivatives_joint_probabilities_sign = None
        if self.prior_mean_beta is None or self.num_regressors != np.size(self.prior_mean_beta):
            if prior_mean_scale is None:
                prior_mean_scale = 0.0
            self.prior_mean_beta = prior_mean_scale * np.ones(self.num_regressors)
        if self.prior_var_beta is None or self.num_regressors != prior_var_beta.shape[0] or self.num_regressors != \
                prior_var_beta.shape[1]:
            if prior_var_scale is None:
                prior_var_scale = 100.0
            self.prior_var_beta = prior_var_scale * np.identity(self.num_regressors)

    def get_intercept_codes(self):
        if self.intercept_grouping is None or self.intercept_grouping == np.array([]):
            self.intercept_codes = ["intercept"]
        else:
            self.num_intercept_groups = self.intercept_grouping.shape[0]
            self.intercept_codes = []
            for g in range(0, self.num_intercept_groups):
                self.intercept_codes.append(("intercept_group_" + str(g)))

    def get_endo_vars(self):
        endo_vars = []
        if self.regular_grid:
            self.lag_length = np.size(self.nbh_sequence)
            for lag in range(0, int(self.lag_length)):
                restriction = self.restriction_sequence[lag]
                nbh = self.nbh_sequence[lag]
                if restriction == 0:
                    if nbh == 0:
                        endo_vars.append(["center"])
                    elif nbh == 4:
                        endo_vars.append(["center", "top", "left", "right",
                                          "bottom"])
                    elif nbh == 8:
                        endo_vars.append(["center",
                                          "top", "left", "right", "bottom",
                                          "topleft", "topright", "bottomleft", "bottomright"])
                elif restriction == 4:
                    if nbh == 0:
                        endo_vars.append(["center"])
                        print("Warning: Restriction sequence")
                        print("contained 4, nbh sequence a 1-nbh")
                        print("at the same position.\n")
                    elif nbh == 4:
                        endo_vars.append(["center", "4_inner_nbh_res"])
                    elif nbh == 8:
                        endo_vars.append(["center", "4_outer_nbh_res",
                                          "4_inner_nbh_res"])
                elif restriction == 8:
                    if nbh == 0:
                        endo_vars.append(["center"])
                        print("Warning: Restriction sequence")
                        print("contained 8, nbh sequence a 1-nbh")
                        print("at the same position.\n")
                    elif nbh == 4:
                        endo_vars.append(["center", "4_inner_nbh_res"])
                        print("Warning: Restriction sequence")
                        print("contained 8, nbh sequence a 4-nbh")
                        print("at the same position.\n")
                    elif nbh == 8:
                        endo_vars.append(["center", "8_nbh_res"])
                        print("Warning: Restriction = 8, which is not fully implemented")
        elif self.general_nbh_coupling == "weak coupling":
            self.lag_length = int(len(self.general_nbh_restriction_sequence))
            self.empty_nbhs = []
            self.sum_empty_nbhs_per_lag = np.zeros(self.lag_length)
            for lag in range(0, int(self.lag_length)):
                new_endo_vars_entry = []
                for location in range(0, self.S1 * self.S2):
                    new_endo_vars_entry.append("general_nbh_" + str(lag) + "_" + "center" + "_" + str(location))
                    self.empty_nbhs.append(False)
                    relevant_nbh_indices = self.general_nbh_restriction_sequence[lag]
                    for nbh_index in relevant_nbh_indices:
                        if nbh_index:
                            if self.general_nbh_sequence[location][nbh_index]:
                                new_endo_vars_entry.append("general_nbh_" +
                                                           str(lag) + "_" + str(nbh_index) + "_" +
                                                           str(location))
                                self.empty_nbhs.append(False)
                            else:
                                self.empty_nbhs.append(True)
                                self.sum_empty_nbhs_per_lag[lag] += 1
                    endo_vars.append(new_endo_vars_entry)
                    new_endo_vars_entry = []
        elif self.general_nbh_coupling == "strong coupling":
            self.lag_length = int(len(self.general_nbh_restriction_sequence))
            for lag in range(0, int(self.lag_length)):
                new_endo_vars_entry = ["general_nbh_" + str(lag) + "_center"]
                relevant_nbh_indices = self.general_nbh_restriction_sequence[lag]
                for nbh_index in relevant_nbh_indices:
                    new_endo_vars_entry.append("general_nbh_" + str(lag) + "_" + str(nbh_index))
                endo_vars.append(new_endo_vars_entry)
        elif (self.general_nbh_coupling is None) and (not self.regular_grid):
            endo_vars = []
            self.lag_length = 0
        self.endo_vars = endo_vars

    def get_lag_counts(self):
        """Only called in __init__: Gets self.lag_counts"""
        self.lag_counts = [self.num_exo_regressors]
        last_count = self.num_exo_regressors
        if self.regular_grid:
            for entry in self.endo_vars:
                self.lag_counts.append(last_count + len(entry) + 1)
                last_count = last_count + len(entry) + 1  # update
        elif self.general_nbh_coupling == "strong coupling":
            for lag in range(0, self.lag_length):
                self.lag_counts.append(last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1)))
                last_count = last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1))
        elif self.general_nbh_coupling == "weak coupling":
            for lag in range(0, self.lag_length):
                self.lag_counts.append(last_count + (
                        (len(self.general_nbh_restriction_sequence[lag]) + 1)
                        * self.S1 * self.S2) - self.sum_empty_nbhs_per_lag[lag])
                last_count = last_count + (- self.sum_empty_nbhs_per_lag[lag] +
                                           (len(self.general_nbh_restriction_sequence[lag]) + 1)
                                           * self.S1 * self.S2)
        elif (not self.regular_grid) and self.general_nbh_coupling is None:
            """STEP 1.D: We only fit a constant, so self.lag_counts remains
            unchanged. self.lag_counts will be None"""

    def get_endo_regressors_per_lag(self):
        """Returns as output the endogeneous regressors per lag"""
        if self.regular_grid:
            """STEP 1A: If we have the 4-nbh structure"""
            endo_regressors_per_lag = []
            for l in range(0, self.lag_length):
                res = self.restriction_sequence[l]
                nbh = self.nbh_sequence[l]
                if res == 0:
                    endo_regressors_per_lag.append(int(nbh) + 1)
                elif res == 4:
                    endo_regressors_per_lag.append(int(nbh * 0.25) + 1)
        elif self.general_nbh_coupling is not None:
            endo_regressors_per_lag = []
            for l in range(0, self.lag_length):
                endo_regressors_per_lag.append(int(len(self.endo_vars[l])))
        else:
            endo_regressors_per_lag = []
        return endo_regressors_per_lag

    def get_extraction_list(self, endo_regressors_per_lag):
        self.extraction_list = [False] * (self.num_exo_regressors)
        if self.regular_grid:
            for i in range(0, self.lag_length - 1):
                self.extraction_list = (self.extraction_list
                                        + [True] * endo_regressors_per_lag[i + 1]
                                        + [False] * int(endo_regressors_per_lag[i] -
                                                        endo_regressors_per_lag[i + 1]))
            self.extraction_list += [False] * endo_regressors_per_lag[self.lag_length - 1]

        elif self.general_nbh_coupling == "weak coupling":
            per_location = []
            for lag in range(0, self.lag_length - 1):
                num_retained = (1 + len(np.intersect1d(
                    self.general_nbh_restriction_sequence[lag],
                    self.general_nbh_restriction_sequence[lag + 1])))
                num_discarded = (-num_retained + 1 +
                                 len(self.general_nbh_restriction_sequence[lag]))
                per_location += ([True] * num_retained +
                                 [False] * num_discarded)
            total_num_last_lag = 1 + len(self.general_nbh_restriction_sequence[self.lag_length - 1])
            per_location += ([False] * total_num_last_lag)
            self.extraction_list += sum([self.S1 * self.S2 * [e] for e in per_location], [])
            self.extraction_list[self.num_exo_regressors:] = np.array(self.extraction_list)[np.where(np.array(self.empty_nbhs) == False)].tolist()

        elif self.general_nbh_coupling == "strong coupling":
            """STEP 1C: IF we have general nbhs"""
            per_location = []
            for lag in range(0, self.lag_length - 1):
                num_retained = (1 + len(np.intersect1d(
                    self.general_nbh_restriction_sequence[lag],
                    self.general_nbh_restriction_sequence[lag + 1])))
                num_discarded = (-num_retained + 1 +
                                 len(self.general_nbh_restriction_sequence[lag]))
                per_location += ([True] * num_retained +
                                 [False] * num_discarded)
            """STEP 2C: The last lag of X_t-1 will 'slide out' of sight, so it 
                       definitely is not needed for X_t anymore."""
            total_num_last_lag = 1 + len(
                self.general_nbh_restriction_sequence[self.lag_length - 1])
            per_location += ([False] * total_num_last_lag)

            """STEP 3C: Use that we have the same structure all across the 
            lattice, and simply multiply each entry of 'per_location' by the
            number of lattice elements"""
            self.extraction_list += per_location

        elif self.general_nbh_coupling is None and not self.regular_grid:
            """We have constant function and don't need to change anything"""

        """STEP 4: In order to copy entries of X_t-1 to X_t, you need to know
                     the position of X_t at which you should insert. (This does
                     only affect the endogeneous part of the regressors)"""
        self.insertion_position = - sum(self.extraction_list)

    def reinstantiate(self, a=None, b=None):
        prior_a, prior_b, S1, S2 = self.a, self.b, self.S1, self.S2
        prior_mean_beta, prior_var_beta = self.prior_mean_beta, self.prior_var_beta
        nbh_sequence = self.nbh_sequence
        restriction_sequence = self.restriction_sequence
        intercept_grouping = self.intercept_grouping
        general_nbh_sequence = self.general_nbh_sequence
        general_nbh_restriction_sequence = self.general_nbh_restriction_sequence
        nbh_sequence_exo = self.nbh_sequence_exo
        exo_selection = self.exo_selection
        padding = self.padding
        auto_prior_update = self.auto_prior_update
        hyperparameter_optimization = self.hyperparameter_optimization
        general_nbh_coupling = self.general_nbh_coupling
        non_spd_alerts = self.non_spd_alerts
        if a is None:
            a = prior_a
        if b is None:
            b = prior_b
        clone_model = BVARNIG(prior_a=a, prior_b=b, S1=S1, S2=S2,
                              prior_mean_beta=prior_mean_beta,
                              prior_var_beta=prior_var_beta,
                              prior_mean_scale=None, prior_var_scale=None,
                              nbh_sequence=nbh_sequence,
                              restriction_sequence=restriction_sequence,
                              intercept_grouping=intercept_grouping,
                              general_nbh_sequence=general_nbh_sequence,
                              general_nbh_restriction_sequence=general_nbh_restriction_sequence,
                              nbh_sequence_exo=nbh_sequence_exo, exo_selection=exo_selection,
                              padding=padding, auto_prior_update=auto_prior_update,
                              hyperparameter_optimization=hyperparameter_optimization,
                              general_nbh_coupling=general_nbh_coupling,
                              non_spd_alerts=non_spd_alerts)
        return clone_model

    def initialization(self, X_endo, X_exo, Y_2, X_exo_2, cp_model, model_prior, padding_columns_computeXX=None, padding_column_get_x_new=None):
        Y1 = X_endo[-1, :].flatten()
        Y2 = Y_2.flatten()
        if self.has_lags:
            X1_endo = X_endo[:self.lag_length, :].reshape(self.lag_length, self.S1, self.S2)
        else:
            X1_endo = None
        if self.exo_bool:
            X1_exo = (X_exo[-1, :, :].reshape(self.num_exo_regressors, self.S1, self.S2))
        else:
            X1_exo = None
        self.XX = np.zeros(shape=(self.num_regressors, self.num_regressors))
        self.XY = np.zeros(self.num_regressors)
        self.X_t = np.zeros(shape=(self.S1 * self.S2, self.num_regressors))
        self.X_tp1 = np.zeros(shape=(self.S1 * self.S2, self.num_regressors))
        self.YY = np.inner(Y1, Y1)
        self.XX_rt = np.zeros(shape=(2, self.num_regressors, self.num_regressors))
        self.XY_rt = np.zeros(shape=(2, self.num_regressors))
        self.YY_rt = np.zeros(2)
        self.Q_rt = np.zeros(shape=(2, self.num_regressors, self.num_regressors))
        self.R_rt = np.zeros(shape=(2, self.num_regressors, self.num_regressors))
        self.M_inv_1_rt = np.zeros(shape=(2, self.num_regressors, self.num_regressors))
        self.M_inv_2_rt = np.zeros(shape=(2, self.num_regressors, self.num_regressors))
        self.log_det_1_rt = np.zeros(2)
        self.log_det_2_rt = np.zeros(2)
        self.beta_XX_beta_rt = np.zeros(2)
        self.beta_rt = np.zeros(shape=(2, self.num_regressors))
        self.retained_run_lengths = np.array([0, 0])
        self.compute_X_XX_XY_YY(Y1, X1_endo, X1_exo, padding_columns_computeXX, compute_XY=True)
        self.X_tp1 = self.get_x_new(Y2, X_exo_2, 1, padding_column_get_x_new)
        self.D_inv = np.linalg.inv(self.prior_var_beta)
        _, self.D_inv_log_det = np.linalg.slogdet(self.D_inv)
        self.D_inv_Q, self.D_inv_R = np.linalg.qr(self.D_inv)
        self.D_inv_log_det = np.sum(np.log(np.abs(np.diagonal(self.D_inv_R))))
        M_inv_1 = np.linalg.inv(self.D_inv + self.XX)
        self.M_inv_1_rt[0, :, :] = self.M_inv_1_rt[1, :, :] = M_inv_1
        Q0, R0 = self.QR_loop(self.D_inv_Q, self.D_inv_R, self.X_t)
        self.Q_rt[0, :, :] = self.Q_rt[1, :, :] = Q0
        self.R_rt[0, :, :] = self.R_rt[1, :, :] = R0
        self.D_inv_b0 = np.matmul(self.D_inv, self.prior_mean_beta)
        self.b0_D_inv_b0 = np.inner(self.prior_mean_beta, self.D_inv_b0)
        self.XX_rt[0, :, :] = self.XX_rt[1, :, :] = self.XX + self.D_inv
        self.XY_rt[0, :] = self.XY_rt[1, :] = (self.XY + self.D_inv_b0)
        self.YY_rt[0] = self.YY_rt[1] = self.YY
        sign, value = np.linalg.slogdet(self.M_inv_1_rt[0, :, :])
        self.log_det_1_rt[0] = self.log_det_1_rt[1] = (value)  # s.p.d. matrices have pos dets
        beta = np.matmul(self.M_inv_1_rt[0, :, :], self.XY_rt[0, :])
        self.beta_rt[0, :] = self.beta_rt[1, :] = beta
        self.beta_XX_beta_rt[0] = self.beta_XX_beta_rt[1] = (np.inner(np.matmul(self.beta_rt[0, :], self.XX_rt[0, :]), self.beta_rt[0, :]))
        small_matrix_inv = np.linalg.inv(np.identity(self.S1 * self.S2) + np.matmul(self.X_tp1, np.matmul(self.M_inv_1_rt[0, :, :], np.transpose(self.X_tp1))))
        sign2, value2 = np.linalg.slogdet(small_matrix_inv)
        self.log_det_2_rt[0] = self.log_det_2_rt[1] = value2 + self.log_det_1_rt[0]
        M_inv_1_x_X_tp1 = np.matmul(self.M_inv_1_rt[0, :, :], np.transpose(self.X_tp1))
        self.M_inv_2_rt[0, :, :] = self.M_inv_2_rt[1, :, :] = self.M_inv_1_rt[0, :, :] - np.matmul(M_inv_1_x_X_tp1, np.matmul(small_matrix_inv, np.transpose(M_inv_1_x_X_tp1)))
        a_ = self.a + 0.5
        b_ = self.b + 0.5 * (self.b0_D_inv_b0 + self.YY - self.beta_XX_beta_rt[0])
        C_0_inv = (a_ / b_) * (np.identity(self.S1 * self.S2) - np.matmul(self.X_t, np.matmul(self.M_inv_1_rt[0, :, :], np.transpose(self.X_t))))
        if b_ < 0:
            log_det = np.nan
        else:
            log_det = ((self.S1 * self.S2) * (np.log(b_) - np.log(a_)) + self.D_inv_log_det - self.log_det_1_rt[0])
        resid = Y1 - np.matmul(self.X_t, self.beta_rt[0, :])
        self.model_log_evidence = np.log(model_prior) + BVARNIG.mvt_log_density(resid, C_0_inv, log_det, 2 * a_, self.non_spd_alerts)
        if cp_model.pmf_0(1) == 0:
            epsilon = 0.000000000001
        else:
            epsilon = 0
        r_equal_0 = self.model_log_evidence + np.log(cp_model.pmf_0(0) + epsilon)
        r_larger_0 = self.model_log_evidence + np.log(cp_model.pmf_0(1) + epsilon)
        self.joint_log_probabilities = np.array([r_equal_0, r_larger_0])
        self.model_specific_joint_log_probabilities_derivative = np.ones((2, 2))
        self.model_specific_joint_log_probabilities_derivative_sign = np.ones((2, 2))
        if self.alpha_rld_learning:
            self.log_alpha_derivatives_joint_probabilities = None
            self.log_alpha_derivatives_joint_probabilities_sign = None

    def compute_X_XX_XY_YY(self, Y0, X0_endo, X0_exo, padding_columns=None, compute_XY=True):
        if self.has_lags:
            X0_endo = X0_endo.reshape(self.lag_length, self.S1, self.S2)
        else:
            X0_endo = None
        lag_count1, lag_count2 = 0, 0
        for i in range(0, self.num_regressors):
            if i <= (self.num_exo_regressors - 1):
                data_vector1 = self.get_exo_regressors(self.all_vars[i], i, X0_exo)
            elif self.has_lags:
                if i >= self.lag_counts[lag_count1]:
                    lag_count1 = lag_count1 + 1
                if padding_columns is None:
                    data_vector1 = self.get_endo_regressors(self.all_vars[i], lag_count1, X0_endo)
                else:
                    data_vector1 = self.get_endo_regressors(self.all_vars[i], lag_count1, X0_endo, padding_columns[i, :])
            lag_count2 = 0
            for j in range(0, self.num_regressors):
                if i <= j:
                    if j <= (self.num_exo_regressors - 1):
                        data_vector2 = self.get_exo_regressors(self.all_vars[j], j, X0_exo)
                    elif self.has_lags:
                        if j >= self.lag_counts[lag_count2]:
                            lag_count2 = lag_count2 + 1
                        if padding_columns is None:
                            data_vector2 = self.get_endo_regressors(
                                self.all_vars[j], lag_count2, X0_endo)
                        else:
                            data_vector2 = self.get_endo_regressors(
                                self.all_vars[j], lag_count2, X0_endo,
                                padding_columns[i, :])
                    if i == 0:
                        self.X_t[:, j] = data_vector2
                        if compute_XY:
                            self.XY[j] = np.inner(data_vector2, Y0)
                    prod = np.inner(data_vector1, data_vector2)
                    self.XX[i, j] = prod
                    self.XX[j, i] = prod
        self.YY = np.inner(Y0, Y0)

    def evaluate_predictive_log_distribution(self, y, t):
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = -np.inf * np.ones(shape=run_length_num)
        self.C_t_inv = np.zeros((run_length_num + 1, self.S1 * self.S2, self.S1 * self.S2))
        self.predictive_variance_log_det = np.zeros(run_length_num + 1)
        self.C_t_inv[0, :, :] = self.C_t_inv_r0
        self.predictive_variance_log_det[0] = self.predictive_variance_r0_log_det
        for r in range(0, run_length_num):
            a_ = self.a + (self.retained_run_lengths[r] + 1.0) * 0.5
            b_ = self.b + 0.5 * (self.b0_D_inv_b0 + self.YY_rt[r] - self.beta_XX_beta_rt[r])
            self.C_t_inv[r + 1, :, :] = (np.identity(self.S1 * self.S2) -
                                         np.matmul(self.X_tp1, np.matmul(self.M_inv_2_rt[r, :, :],
                                                                         np.transpose(self.X_tp1))))
            if b_ < 0:
                log_det = np.nan
            else:
                log_det = ((self.S1 * self.S2) * (np.log(b_) - np.log(a_)) +
                           self.log_det_1_rt[r] - self.log_det_2_rt[r])
            self.predictive_variance_log_det[r + 1] = log_det

            """STEP 2.3: Evaluate the predictive probability for r_t = r"""
            resid = y - np.matmul(self.X_tp1, self.beta_rt[r, :])
            log_densities[r] = (
                BVARNIG.mvt_log_density(resid,
                                        (a_ / b_) * self.C_t_inv[r + 1, :, :],
                                        log_det, 2 * a_, self.non_spd_alerts))

        """STEP 3: return the full log density vector"""
        return log_densities

    def get_log_integrals_power_divergence(self):
        """get integrals for power div in log-form"""
        p = self.S1 * self.S2
        run_length_with_0 = np.insert(self.retained_run_lengths.copy() + 1, 0, 0)

        nu_1 = 2 * (self.a + (run_length_with_0 + 1.0) * 0.5)
        nu_2 = nu_1 * self.alpha_rld + p * self.alpha_rld + nu_1

        C1 = (1.0 + self.alpha_rld) * (special.gammaln(0.5 * (nu_1 + p)) -
                                       special.gammaln(0.5 * nu_1))
        C2 = (special.gammaln(0.5 * (nu_2 + p)) - special.gammaln(0.5 * nu_2))

        return (C1 - C2 - nu_1 * 0.5 * p * self.alpha_rld
                - np.pi * 0.5 * p * self.alpha_rld  # dets)
                - self.alpha_rld * self.predictive_variance_log_det)

    def evaluate_log_prior_predictive(self, y, t):
        resid = y - np.matmul(self.X_tp1, self.prior_mean_beta)
        self.C_t_inv_r0 = (
                np.identity(self.S1 * self.S2) -
                np.matmul(self.X_tp1, np.matmul(self.prior_var_beta,
                                                np.transpose(self.X_tp1))))
        _, log_det = np.linalg.slogdet((self.a / self.b) * self.C_t_inv_r0)
        self.predictive_variance_r0_log_det = log_det
        return min(0.0, BVARNIG.mvt_log_density(resid,
                                                (self.a / self.b) * self.C_t_inv_r0, log_det, 2 * self.a, True))

    def save_NLL_fixed_pars(self, y, t):
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = -np.inf * np.ones(shape=run_length_num)
        for r in range(0, run_length_num):
            a_ = self.a + (self.retained_run_lengths[r] + 1.0) * 0.5
            b_ = (self.b + 0.5 * (self.b0_D_inv_b0 + self.YY_rt[r] -
                                  self.beta_XX_beta_rt[r]))
            sigma2 = max((b_ / (a_ + 1)), 0.0000001)
            cov_mat = sigma2 * self.C_t_inv[r + 1, :, :]
            resid = y - np.matmul(self.X_tp1, self.beta_rt[r, :])
            log_densities[r] = (
                stats.multivariate_normal.logpdf(resid, cov=cov_mat))
        self.one_step_ahead_predictive_log_probs_fixed_pars = log_densities

    def update_predictive_distributions(self, y_t, y_tm1, x_exo_t, x_exo_tp1, t,
                                        padding_column_tm1=None,
                                        padding_column_t=None,
                                        r_evaluations=None):
        y_t, y_tm1 = y_t.flatten(), y_tm1.flatten()
        self.XX_old, self.XY_old, self.X_t_old = (self.XX.copy(),
                                                  self.XY.copy(), self.X_t.copy())
        self.Y_new, self.Y_old = y_t, y_tm1

        """STEP 2.2: Updates X'X, X'Y, Y'Y, XX_rt, XY_rt, YY_rt"""
        self.regressor_cross_product_updates(y_t, y_tm1, x_exo_t,
                                             t, padding_column_tm1)
        self.X_tp1 = self.get_x_new(y_t, x_exo_tp1, t, padding_column_t)
        self.pre_updates(t)
        self.updates(t)

    def regressor_cross_product_updates(self, y_t, y_tm1, x_exo, t,
                                        padding_column=None,
                                        rt_updates=True):
        if self.has_lags and self.lag_length > 1:
            self.X_t[:, self.insertion_position:] = (
                self.X_t[:, self.extraction_list])
            self.XX[self.insertion_position:, self.insertion_position:] = (
                self.XX[self.extraction_list, :][:, self.extraction_list])
        i = 0
        if (not (self.restriction_sequence is None) or
                self.general_nbh_coupling == "strong coupling"):
            num_new_vars = len(self.endo_vars[0]) + self.num_exo_regressors
            new_vars = sum(self.exo_vars, []) + self.endo_vars[0]
        elif self.general_nbh_coupling == "weak coupling":
            new_vars = (sum(self.exo_vars, []) +
                        sum(self.endo_vars[:self.S1 * self.S2], []))
            num_new_vars = int(len(new_vars))
        elif self.general_nbh_coupling is None and not self.regular_grid:
            new_vars = sum(self.exo_vars, [])
            num_new_vars = int(len(new_vars))
        for regressor_code in new_vars:  # sum(self.exo_vars,[]) + self.endo_vars[0]:
            if i <= self.num_exo_regressors - 1:
                x_i = self.get_exo_regressors(regressor_code, i, x_exo)
            elif self.has_lags:
                x_i = self.get_endo_regressors(regressor_code, 1,
                                               y_tm1.reshape(1, self.S1, self.S2),
                                               padding_column)
            self.X_t[:, i] = x_i
            for j in range(0, num_new_vars):
                if (i <= j):
                    if (j <= self.num_exo_regressors - 1):
                        x_j = self.get_exo_regressors(self.all_vars[j],
                                                      j, x_exo)
                    elif self.has_lags:
                        x_j = self.get_endo_regressors(self.all_vars[j],
                                                       1, y_tm1.reshape(1, self.S1, self.S2),
                                                       padding_column)
                    self.XX[i, j] = self.XX[j, i] = np.inner(x_i, x_j)
                if i == 0 and self.has_lags:
                    for k in range(num_new_vars, self.num_regressors):
                        x_k = self.X_t[:, k]
                        self.XX[k, j] = self.XX[j, k] = np.inner(x_j, x_k)
            i = i + 1
        if not self.has_lags:
            self.XX = np.identity(self.num_regressors)
        self.YY = np.inner(y_t, y_t)
        self.XY = np.matmul(np.transpose(self.X_t), y_t)
        if rt_updates:
            self.XX_rt = self.XX_rt + self.XX
            self.XY_rt = self.XY_rt + self.XY
            self.YY_rt = self.YY_rt + self.YY
            self.XX_rt = np.insert(self.XX_rt, 0, (self.XX + self.D_inv), axis=0)
            self.XY_rt = np.insert(self.XY_rt, 0, (self.XY + self.D_inv_b0), axis=0)
            self.YY_rt = np.insert(self.YY_rt, 0, self.YY, axis=0)

    def get_x_new(self, y_t, x_exo_tp1, t, padding_column=None):
        if self.has_lags and self.lag_length > 1:
            x_new = np.zeros((self.S1 * self.S2, self.num_regressors))
            x_new[:, self.insertion_position:] = (
                self.X_t[:, self.extraction_list].copy())
        else:
            x_new = np.zeros((self.S1 * self.S2, self.num_regressors))
        i = 0
        if self.has_lags:
            all_codes = sum(self.exo_vars, []) + self.endo_vars[0]
        else:
            all_codes = sum(self.exo_vars, [])
        for regressor_code in all_codes:
            if i <= self.num_exo_regressors - 1:
                x_i = self.get_exo_regressors(regressor_code, i, x_exo_tp1)
            elif self.has_lags:
                x_i = self.get_endo_regressors(regressor_code, 1,
                                               y_t.reshape(1, self.S1, self.S2),
                                               padding_column)
            x_new[:, i] = x_i
            i = i + 1
        return x_new

    def pre_updates(self, t):
        self.retained_run_lengths = self.retained_run_lengths + 1
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)
        new_M_inv = np.linalg.inv(self.D_inv + self.XX)
        self.M_inv_1_rt = np.insert(self.M_inv_2_rt.copy(), 0, new_M_inv, axis=0)
        self.compute_betas(t)
        sign, new_log_det = np.linalg.slogdet(new_M_inv)
        self.log_det_1_rt = np.insert(self.log_det_2_rt.copy(), 0, new_log_det)

    def updates(self, t):
        run_length_num = self.retained_run_lengths.shape[0]
        self.M_inv_2_rt = np.zeros((run_length_num, self.num_regressors, self.num_regressors))
        self.log_det_2_rt = np.zeros(run_length_num)
        for r in range(0, run_length_num):
            M_inv_x_X_tp1 = np.matmul(self.M_inv_1_rt[r, :, :], np.transpose(self.X_tp1))
            small_matrix_inv = np.linalg.inv(np.identity(self.S1 * self.S2) +
                                             np.matmul(np.transpose(M_inv_x_X_tp1),
                                                       np.transpose(self.X_tp1)))
            self.M_inv_2_rt[r, :, :] = self.M_inv_1_rt[r, :, :] - np.matmul(
                (M_inv_x_X_tp1), np.matmul(small_matrix_inv,
                                           np.transpose(M_inv_x_X_tp1)))
            sign, value = np.linalg.slogdet(small_matrix_inv)
            self.log_det_2_rt[r] = value + self.log_det_1_rt[r]

    def post_QR_updates(self, t):
        self.M_inv_2_rt = np.insert(self.M_inv_2_rt, 0,
                                    np.zeros((self.num_regressors, self.num_regressors)),
                                    axis=0)
        run_length_num = self.retained_run_lengths.shape[0]
        for r in range(0, run_length_num):
            self.M_inv_2_rt[r, :, :] = linalg.solve_triangular(a=self.R_rt[r, :, :],
                                                               b=np.transpose(self.Q_rt[r, :, :]), check_finite=False)
        self.log_det_2_rt = np.sum(np.log(np.abs(np.diagonal(self.R_rt,
                                                             axis1=1, axis2=2))), axis=1)

    def compute_betas(self, t):
        run_length_num = self.retained_run_lengths.shape[0]
        self.beta_rt = (
            np.insert(self.beta_rt, 0, np.zeros(self.num_regressors), axis=0))
        self.beta_XX_beta_rt = np.insert(self.beta_XX_beta_rt, 0, 0, axis=0)
        for r in range(0, run_length_num):
            self.beta_rt[r, :] = np.matmul(self.M_inv_1_rt[r, :, :], self.XY_rt[r, :])
            self.beta_XX_beta_rt[r] = np.inner(self.beta_rt[r, :], np.matmul(self.XX_rt[r, :, :], self.beta_rt[r, :]))

    def QR_loop(self, Q0, R0, X):
        current_count = end_point = 0
        while (end_point != self.S1 * self.S2):
            start_point = current_count * self.num_regressors
            end_point = min((current_count + 1) * self.num_regressors,
                            self.S1 * self.S2)
            current_range = range(start_point, end_point)
            Q0, R0 = linalg.qr_update(Q0, R0,
                                      np.transpose(X[current_range, :]),
                                      np.transpose(X[current_range, :]),
                                      check_finite=False)
            current_count = current_count + 1
        return Q0, R0

    def get_posterior_expectation(self, t, r_list=None):
        post_mean = np.matmul(self.X_tp1, self.beta_rt[:, :, np.newaxis])
        return post_mean

    def get_prior_expectation(self, t):
        return np.matmul(self.X_tp1, self.prior_mean_beta)

    def get_posterior_variance(self, t, r_list=None):
        post_var = np.zeros((np.size(self.retained_run_lengths), self.S1 * self.S2, self.S1 * self.S2))
        run_length_num = self.retained_run_lengths.shape[0]
        for r in range(0, run_length_num):
            a_ = self.a + (r + 1.0) * 0.5
            b_ = (self.b + 0.5 * (self.b0_D_inv_b0 + self.YY_rt[r] -
                                  self.beta_XX_beta_rt[r]))
            post_var[r, :, :] = (b_ / a_) * (np.identity(self.S1 * self.S2) +
                                             np.matmul(self.X_tp1, np.matmul(self.M_inv_1_rt[r, :, :],
                                                                             np.transpose(self.X_tp1))))
        return post_var

    @staticmethod
    def mvt_log_density(y_flat, prec, log_det, df, prior=False, alerts=False):
        p, nu = y_flat.shape[0], df
        log_term = (1 + (1.0 / nu) * np.matmul(np.matmul(y_flat, prec), y_flat))
        if (log_term < 0 or np.isnan(log_det)):
            if not prior and p > 1:
                if alerts:
                    print("covariance estimate not s.p.d. or log_det nan")
                    print("degrees of freedom: ", df)
                try:
                    prec = (NPD.nearestPD(prec) +
                            np.identity(prec.shape[0]) * max(df * nu, max(25, p)))
                except (ValueError, np.linalg.LinAlgError) as e:  # np.linalg.LinAlgError
                    prec = prec + np.identity(p) * pow(10, 5)
                log_term = (1 + (1.0 / nu) *
                            np.matmul(np.matmul(y_flat, prec), y_flat))
            elif prior and p > 1:
                if log_term < 0 and p > 1:
                    prec = prec + np.identity(prec.shape[0]) * nu * df
                    log_term = (1 + (1.0 / nu) * np.matmul(
                        np.matmul(y_flat, prec), y_flat))
                if log_term < 0 and p > 1:
                    prec = NPD.nearestPD(prec) + np.identity(prec.shape[0]) * max(df * nu, 25)
                    log_term = (1 + (1.0 / nu) * np.matmul(np.matmul(y_flat, prec), y_flat))
                count = 0
                while log_term < 0:
                    if count == 0:
                        print("Covariance matrix injected with sphericity")
                    prec = prec + np.identity(prec.shape[0]) * nu * df * 10
                    log_term = (1 + (1.0 / nu) * np.matmul(np.matmul(y_flat, prec), y_flat))
                    count = count + 1

            elif p == 1:
                """If we only fit a single constant!"""
                return -pow(10, 4)
            if (log_term < 0):
                print("non-s.p.d. covariance estimate:", "problem persists! Set it to log(pow(10,-100))")
                print("log term is", log_term)
                print("det term is", log_det)
                return -pow(10, 5)
            else:
                log_term = np.log(log_term)
                _, log_det = np.linalg.slogdet(prec)
            if np.isnan(log_det):
                print("log_det nan: problem persists!")
        else:
            log_term = np.log(log_term)
        if np.isnan(log_det):
            print("nan log det")
            _, log_det = np.linalg.slogdet(prec)
            log_det = 1.0 / log_det
            if np.isnan(log_det):
                print("problem persists!")
        calc = (special.gammaln(0.5 * (nu + p)) - special.gammaln(0.5 * nu) -
                0.5 * p * (np.log(nu) + np.log(np.pi)) - 0.5 * log_det -
                0.5 * (nu + p) * log_term)
        if np.isnan(calc):
            print("Alert! Calc is nan")
            calc = -pow(10, 5)
        return calc

    def trimmer(self, kept_run_lengths, BAR_submodel=False):
        if not BAR_submodel:
            self.joint_log_probabilities = self.joint_log_probabilities[kept_run_lengths]
        if self.hyperparameter_optimization:
            self.model_specific_joint_log_probabilities_derivative_sign = (
                self.model_specific_joint_log_probabilities_derivative_sign[:,
                kept_run_lengths])
            self.model_specific_joint_log_probabilities_derivative = (
                self.model_specific_joint_log_probabilities_derivative[:,
                kept_run_lengths])
        if self.generalized_bayes_rld == "power_divergence" and self.alpha_rld_learning and self.log_alpha_derivatives_joint_probabilities is not None:
            self.log_alpha_derivatives_joint_probabilities = self.log_alpha_derivatives_joint_probabilities[
                kept_run_lengths]
            self.log_alpha_derivatives_joint_probabilities_sign = self.log_alpha_derivatives_joint_probabilities_sign[
                kept_run_lengths]
        self.beta_rt = self.beta_rt[kept_run_lengths, :]
        self.beta_XX_beta_rt = self.beta_XX_beta_rt[kept_run_lengths]
        self.XX_rt = self.XX_rt[kept_run_lengths, :, :]
        self.XY_rt = self.XY_rt[kept_run_lengths, :]
        self.YY_rt = self.YY_rt[kept_run_lengths]
        self.M_inv_1_rt = self.M_inv_1_rt[kept_run_lengths, :, :]
        self.M_inv_2_rt = self.M_inv_2_rt[kept_run_lengths, :, :]
        self.log_det_1_rt = self.log_det_1_rt[kept_run_lengths]
        self.log_det_2_rt = self.log_det_2_rt[kept_run_lengths]
        self.retained_run_lengths = self.retained_run_lengths[kept_run_lengths]
        self.model_log_evidence = scipy.special.logsumexp(self.joint_log_probabilities)

    def get_exo_regressors(self, regressor_code, i, data):
        if regressor_code == "intercept":
            data_vector = np.ones((self.S1, self.S2))
        elif self.intercept_codes != ["intercept"]:
            group_number = int(regressor_code.split("_")[-1])
            data_vector = self.intercept_grouping[group_number, :, :].flatten()
        else:
            data_vector = data[i, :, :].flatten()
        return data_vector.flatten()

    def get_endo_regressors(self, regressor_code, lag, data, padding_column=None):
        padding = self.padding
        lag = -(lag - 1)
        if padding == 0 or padding == "zero":
            padding_row = np.zeros(self.S2)
            padding_col = np.zeros(self.S1)
            padding_corners = 0.0
        elif padding == "overall_mean":
            mean = np.mean(data[lag, :, :])
            padding_row = mean * np.ones(self.S2)
            padding_col = mean * np.ones(self.S1)
            padding_corners = mean
        elif padding == "row_col_mean":
            padding_row = np.mean(data[lag, :, :], axis=0)
            padding_col = np.mean(data[lag, :, :], axis=1)
            weight = (np.size(padding_row) /
                      (np.size(padding_row) + np.size(padding_col)))
            padding_corners = (weight * np.sum(padding_row) +
                               (1 - weight) * np.sum(padding_col))

        elif padding.split("_")[-1] == "rhs" or padding.split("_")[-1] == "lhs":
            """I.e., if we have a CSurf object, we need some extra care at the 
            boundaries of the change surface"""
            padding_row = np.mean(data[lag, :, :], axis=0)

            if padding.split("_")[-1] == "rhs":
                """get padding for cols as usual + specific one for rhs, lhs"""
                padding_rhs = padding_column
                padding_lhs = padding_col = np.mean(data[lag, :, :], axis=1)
                weight = (np.size(padding_row) /
                          (np.size(padding_row) + np.size(padding_col)))
                padding_corner_rhs = (weight * np.sum(padding_row) +
                                      (1 - weight) * np.sum(padding_rhs))
                padding_corner_lhs = padding_corners = (
                        weight * np.sum(padding_row) +
                        (1 - weight) * np.sum(padding_lhs))
            else:
                """get padding for cols as usual + specific one for rhs, lhs"""
                padding_rhs = padding_col = np.mean(data[lag, :, :], axis=1)
                padding_lhs = padding_column
                weight = (np.size(padding_row) /
                          (np.size(padding_row) + np.size(padding_col)))
                padding_corner_rhs = padding_corners = (weight *
                                                        np.sum(padding_row) + (1 - weight) * np.sum(padding_rhs))
                padding_corner_lhs = weight * np.sum(padding_row) + (1 - weight) * np.sum(padding_lhs)
        if regressor_code == "intercept":
            data_vector = np.ones((self.S1, self.S2))
        elif regressor_code == "center":
            data_vector = data[lag, :, :]
        elif regressor_code == "left":
            if padding.split("_")[-1] == "rhs":
                """Insert the padding column passed to this function"""
                data_vector = np.insert(data[lag, :, :-1], 0, padding_rhs, axis=1)
            else:
                """Take the row averages as padding"""
                data_vector = np.insert(data[lag, :, :-1], 0, padding_col, axis=1)

        elif regressor_code == "right":
            if padding.split("_")[-1] == "lhs":
                """Insert the padding column passed to this function"""
                data_vector = np.insert(data[lag, :, 1:], self.S2 - 1, padding_lhs, axis=1)
            else:
                """Take the row averages as padding"""
                data_vector = np.insert(data[lag, :, 1:], self.S2 - 1, padding_col, axis=1)

        elif regressor_code == "top":
            data_vector = np.insert(data[lag, :-1, :], 0, padding_row, axis=0)
        elif regressor_code == "bottom":
            data_vector = np.insert(data[lag, 1:, :], self.S1 - 1, padding_row, axis=0)

        elif regressor_code == "topleft":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[1:, 1:] = data[lag, :-1, :-1]
            if padding.split("_")[-1] == "rhs":
                """Insert the padding column passed to this function"""
                data_vector[0, :] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:, 0] = np.append(padding_corner_rhs, padding_rhs[:-1])
            else:
                """Take the row averages as padding"""
                data_vector[0, :] = np.append(padding_corners, padding_row[:-1])
                data_vector[:, 0] = np.append(padding_corners, padding_col[:-1])

        elif regressor_code == "topright":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[1:, :-1] = data[lag, :-1, 1:]
            if padding.split("_")[-1] == "lhs":
                """Insert the padding column passed to this function"""
                data_vector[0, :] = np.append(padding_row[1:], padding_corner_lhs)
                data_vector[:, -1] = np.append(padding_corner_lhs, padding_lhs[:-1])
            else:
                """Take the row averages as padding"""
                data_vector[0, :] = np.append(padding_row[1:], padding_corners)
                data_vector[:, -1] = np.append(padding_corners, padding_col[:-1])

        elif regressor_code == "bottomleft":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[:-1, 1:] = data[lag, 1:, :-1]
            if padding.split("_")[-1] == "rhs":
                """Insert the padding column passed to this function"""
                data_vector[-1, :] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:, 0] = np.append(padding_rhs[1:], padding_corner_rhs)
            else:
                """Take the row averages as padding"""
                data_vector[-1, :] = np.append(padding_corners, padding_row[:-1])
                data_vector[:, 0] = np.append(padding_col[1:], padding_corners)

        elif regressor_code == "bottomright":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[:-1, :-1] = data[lag, 1:, 1:]
            if padding.split("_")[-1] == "lhs":
                """Insert the padding column passed to this function"""
                data_vector[-1, :] = np.append(padding_row[1:], padding_corner_lhs)
                data_vector[:, -1] = np.append(padding_lhs[1:], padding_corner_lhs)
            else:
                """Take the row averages as padding"""
                data_vector[-1, :] = np.append(padding_row[1:], padding_corners)
                data_vector[:, -1] = np.append(padding_col[1:], padding_corners)

        elif regressor_code == "4_inner_nbh_res":
            if padding.split("_")[-1] == "lhs":
                # pad with real data on the right
                data_vector = (np.insert(data[lag, :, :-1], 0, padding_col, axis=1) +
                               np.insert(data[lag, :, 1:], self.S2 - 1, padding_lhs, axis=1) +
                               np.insert(data[lag, :-1, :], 0, padding_row, axis=0) +
                               np.insert(data[lag, 1:, :], self.S1 - 1, padding_row, axis=0))
            elif padding.split("_")[-1] == "rhs":
                data_vector = (np.insert(data[lag, :, :-1], 0, padding_rhs, axis=1) +
                               np.insert(data[lag, :, 1:], self.S2 - 1, padding_col, axis=1) +
                               np.insert(data[lag, :-1, :], 0, padding_row, axis=0) +
                               np.insert(data[lag, 1:, :], self.S1 - 1, padding_row, axis=0))
            else:
                data_vector = (np.insert(data[lag, :, :-1], 0, padding_col, axis=1) +
                               np.insert(data[lag, :, 1:], self.S2 - 1, padding_col, axis=1) +
                               np.insert(data[lag, :-1, :], 0, padding_row, axis=0) +
                               np.insert(data[lag, 1:, :], self.S1 - 1, padding_row, axis=0))

        elif ((regressor_code == "4_outer_nbh_res") or
              (regressor_code == "8_nbh_res")):
            if padding.split("_")[-1] == "lhs":
                # pad with real data on the right
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag, :-1, :-1]
                data_vector[0, :] = np.append(padding_corners, padding_row[:-1])
                data_vector[:, 0] = np.append(padding_corners, padding_col[:-1])
                """add topright"""
                data_vector[1:, :-1] += data[lag, :-1, 1:]
                data_vector[0, :] += np.append(padding_row[1:], padding_corner_lhs)
                data_vector[:, -1] += np.append(padding_corner_lhs, padding_lhs[:-1])
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag, 1:, :-1]
                data_vector[-1, :] += np.append(padding_corners, padding_row[:-1])
                data_vector[:, 0] += np.append(padding_col[1:], padding_corners)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag, 1:, 1:]
                data_vector[-1, :] += np.append(padding_row[1:], padding_corner_lhs)
                data_vector[:, -1] += np.append(padding_lhs[1:], padding_corner_lhs)
            elif padding.split("_")[-1] == "rhs":
                # pad with real data on the left
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag, :-1, :-1]
                data_vector[0, :] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:, 0] = np.append(padding_corner_rhs, padding_rhs[:-1])
                """add topright"""
                data_vector[1:, :-1] += data[lag, :-1, 1:]
                data_vector[0, :] += np.append(padding_row[1:], padding_corners)
                data_vector[:, -1] += np.append(padding_corners, padding_col[:-1])
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag, 1:, :-1]
                data_vector[-1, :] += np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:, 0] += np.append(padding_rhs[1:], padding_corner_rhs)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag, 1:, 1:]
                data_vector[-1, :] += np.append(padding_row[1:], padding_corners)
                data_vector[:, -1] += np.append(padding_col[1:], padding_corners)
            else:
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag, :-1, :-1]
                data_vector[0, :] = np.append(padding_corners, padding_row[:-1])
                data_vector[:, 0] = np.append(padding_corners, padding_col[:-1])
                """add topright"""
                data_vector[1:, :-1] += data[lag, :-1, 1:]
                data_vector[0, :] += np.append(padding_row[1:], padding_corners)
                data_vector[:, -1] += np.append(padding_corners, padding_col[:-1])
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag, 1:, :-1]
                data_vector[-1, :] += np.append(padding_corners, padding_row[:-1])
                data_vector[:, 0] += np.append(padding_col[1:], padding_corners)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag, 1:, 1:]
                data_vector[-1, :] += np.append(padding_row[1:], padding_corners)
                data_vector[:, -1] += np.append(padding_col[1:], padding_corners)

        if regressor_code == "8_nbh_res":
            if padding.split("_")[-1] == "lhs":
                data_vector += (np.insert(data[lag, :, :-1], 0, padding_col, axis=1) +
                                np.insert(data[lag, :, 1:], self.S2 - 1, padding_lhs, axis=1) +
                                np.insert(data[lag, :-1, :], 0, padding_row, axis=0) +
                                np.insert(data[lag, 1:, :], self.S1 - 1, padding_row, axis=0))
            elif padding.split("_")[-1] == "rhs":
                data_vector += (np.insert(data[lag, :, :-1], 0, padding_rhs, axis=1) +
                                np.insert(data[lag, :, 1:], self.S2 - 1, padding_col, axis=1) +
                                np.insert(data[lag, :-1, :], 0, padding_row, axis=0) +
                                np.insert(data[lag, 1:, :], self.S1 - 1, padding_row, axis=0))
            else:
                data_vector += (np.insert(data[lag, :, :-1], 0, padding_col, axis=1) +
                                np.insert(data[lag, :, 1:], self.S2 - 1, padding_col, axis=1) +
                                np.insert(data[lag, :-1, :], 0, padding_row, axis=0) +
                                np.insert(data[lag, 1:, :], self.S1 - 1, padding_row, axis=0))
        if regressor_code.split("_")[0] == "general":
            data_vector = np.zeros((self.S1, self.S2))
            if self.general_nbh_coupling == "strong coupling":
                num_group = (regressor_code.split("_")[-1])
                if num_group == "center":
                    data_vector = data[lag, :, :].flatten()
                else:
                    num_group = int(num_group)
                    relevant_nbhs = [item[num_group] for item in self.general_nbh_sequence]
                    dat = data[lag, :, :].flatten()
                    data_vector = np.array([sum(dat[relevant_nbh]) for relevant_nbh in relevant_nbhs])
            elif self.general_nbh_coupling == "weak coupling":
                data_vector = data_vector.flatten()
                if regressor_code.split("_")[-2] == "center":
                    location = int(regressor_code.split("_")[-1])
                    data_vector[location] = (data[lag, :, :].flatten()[location])
                else:
                    nbh_index = int(regressor_code.split("_")[-2])
                    lag2 = int(regressor_code.split("_")[-3])
                    relevant_nbh_index = self.general_nbh_restriction_sequence[lag2][nbh_index]
                    location = int(regressor_code.split("_")[-1])
                    relevant_nbh = (self.general_nbh_sequence[location][relevant_nbh_index])
                    data_vector = np.zeros(self.S1 * self.S2)
                    if relevant_nbh:
                        data_vector[location] = sum(data[lag, :, :].flatten()[relevant_nbh])
        return data_vector.flatten()

    def prior_update(self, t, model_specific_rld):
        filled_in = model_specific_rld > -np.inf
        if True:
            self.prior_mean_beta = np.sum(self.beta_rt * np.exp(model_specific_rld[filled_in, np.newaxis]), axis=0)
            self.D_inv_b0 = np.matmul(self.D_inv, self.prior_mean_beta)
            self.b0_D_inv_b0 = np.inner(self.prior_mean_beta, self.D_inv_b0)
        if True:
            a_vec = self.a + (self.retained_run_lengths + 1.0) * 0.5
            b_vec = (self.b + 0.5 * (self.b0_D_inv_b0 + self.YY_rt + self.beta_XX_beta_rt))
            self.a = np.inner(a_vec, np.exp(model_specific_rld[filled_in]))
            self.b = np.inner(b_vec, np.exp(model_specific_rld[filled_in]))

    def get_param_rt(self):
        a_rt = np.insert(self.a + (self.retained_run_lengths + 1.0) * 0.5, 0, self.a)
        b_rt = np.insert(self.b + 0.5 * (self.b0_D_inv_b0 + self.YY_rt - self.beta_XX_beta_rt), 0, self.b)
        return a_rt, b_rt

    def differentiate_predictive_log_distribution(self, y, t, run_length_log_distro):
        y = y.flatten()
        p = np.size(y)
        run_length_num = self.retained_run_lengths.shape[0]
        num_params = 2
        a_vec, b_vec = self.get_param_rt()
        resids = np.insert(np.matmul(self.X_t, self.beta_rt[:, :, np.newaxis]) -
                           y.flatten()[:, np.newaxis], 0,
                           np.matmul(self.X_t, self.prior_mean_beta[:, np.newaxis]) -
                           y.flatten()[:, np.newaxis], axis=0)
        products = np.sum(resids * np.matmul(self.C_t_inv, resids), axis=1).reshape(run_length_num + 1)
        expr1 = scipy.special.psi(a_vec + 0.5 * p)
        expr2 = -scipy.special.psi(a_vec)
        expr3 = -p / (2.0 * a_vec)
        expr5 = 0.5 * self.S1 * self.S2 * (1.0 / a_vec)
        inside6 = (1.0 + 0.5 * (1 / b_vec) * products)
        expr6 = -np.log(np.abs(inside6))
        log_posterior_predictive_gradients_a_val = expr1 + expr2 + expr3 + expr5 + expr6
        expr5_ = -0.5 * self.S1 * self.S2 * (1.0 / b_vec)
        expr6_ = (0.5 * (p + 2 * a_vec)) * (1 / (b_vec * b_vec)) * (0.5 * products / (1 + 0.5 * (1 / b_vec) * products))
        log_posterior_predictive_gradients_b_val = expr5_ + expr6_
        log_gradients_a_sign = np.sign(inside6) * np.sign(log_posterior_predictive_gradients_a_val)
        log_gradients_b_sign = np.sign(log_posterior_predictive_gradients_b_val)
        all_predictives = np.insert(self.one_step_ahead_predictive_log_loss, 0, self.r0_log_loss)
        log_gradients_a_val = (np.log(
            np.max(
                np.array([
                    np.abs(log_posterior_predictive_gradients_a_val),
                    0.000005 * np.ones((run_length_num + 1))]
                ), axis=0)
        ) + all_predictives)
        log_gradients_b_val = (np.log(
            np.max(
                np.array([
                    np.abs(log_posterior_predictive_gradients_b_val),
                    0.000005 * np.ones((run_length_num + 1))]
                ), axis=0)
        ) + all_predictives)
        log_gradients_sign = np.array([log_gradients_a_sign, log_gradients_b_sign]).reshape(num_params,
                                                                                            run_length_num + 1)
        log_gradients_val = np.array([log_gradients_a_val, log_gradients_b_val]).reshape(num_params, run_length_num + 1)
        return log_gradients_val, log_gradients_sign

    def caron_hyperparameter_optimization(self, t, gradient, step_size):
        max_step_scale = 1
        min_step_scale = pow(10, -5)
        disturbance = 0.0
        dif_old_a = self.a - self.a_old + disturbance
        dif_old_b = self.b - self.b_old + disturbance
        dif_old_val = np.array([dif_old_a, dif_old_b])
        dif_old_grad = gradient - self.gradient_old
        dampener = pow((1 / 1.005), t * 0.1)
        D1_sign = np.dot(np.sign(dif_old_val), np.sign(dif_old_grad))
        if D1_sign > 0:
            D1 = min(np.abs(np.dot(dif_old_val, dif_old_grad)), pow(10, 5))
        else:
            D1 = max(- np.abs(np.dot(dif_old_val, dif_old_grad)), -pow(10, 5))
        if np.abs(D1) < pow(10, -1):
            D1 = (pow(10, -1)) * D1_sign
        D2 = min(np.dot(dif_old_grad, dif_old_grad), pow(10, 5))
        if D2 < pow(10, -1):
            D2 = pow(10, -1)
        D3 = min(np.dot(dif_old_val, dif_old_val), pow(10, 5))
        if D3 < pow(10, -1):
            D3 = pow(10, -1)
        alpha_1 = ((D1 / D2) * dampener)
        step_size_abs = max(np.abs(alpha_1), step_size)
        step_size = np.sign(alpha_1) * step_size_abs
        if np.sign(alpha_1) == 0.0:
            sign_a = np.sign(gradient[0])
            sign_b = np.sign(gradient[1])
        else:
            sign_a = np.sign(step_size * gradient[0])
            sign_b = np.sign(step_size * gradient[1])

        dampener = 1.0 / t
        increment_a = (sign_a *
                       max(self.a * dampener * min_step_scale,
                           min(self.a * dampener * max_step_scale,
                               np.abs(step_size_abs * gradient[0]))))
        increment_b = (sign_b *
                       max(self.a * dampener * min_step_scale,
                           min(self.b * dampener * max_step_scale,
                               np.abs(step_size_abs * gradient[1]))))
        dampener = 1.0 / t
        self.a_old = self.a
        self.b_old = self.b
        self.gradient_old = gradient
        self.a = min(max(self.a + increment_a, 1.0), pow(10, pow(10, 3)))
        self.b = min(max(self.b + increment_b, pow(10, -10)), pow(10, pow(10, 3)))
        self.a_list.append(self.a)
        self.b_list.append(self.b)

    def get_one_step_ahead_log_loss_derivatives_power_divergence(self):
        p = self.S1 * self.S2
        run_length_num = np.size(self.retained_run_lengths) + 1
        a_vec, _ = self.get_param_rt()
        nu_1_vec = np.maximum(2 * a_vec, 1.005)
        nu_2_vec = np.maximum(nu_1_vec * p + nu_1_vec * self.alpha_rld + nu_1_vec, 1.005)
        f_1_log = (1.0 + self.alpha_rld) * (special.gammaln(0.5 * (nu_1_vec + p)) - special.gammaln(0.5 * nu_1_vec))
        f_2_log = special.gammaln(0.5 * (nu_2_vec)) - special.gammaln(0.5 * nu_2_vec + p)
        f_3_log = ((np.log(nu_1_vec) + np.log(np.pi)) * (-0.5 * p * self.alpha_rld) +
                   (-self.alpha_rld) * self.predictive_variance_log_det)
        expr_1 = scipy.special.gammaln(0.5 * (nu_1_vec + p)) - scipy.special.gammaln(0.5 * nu_1_vec)
        f_1_der_sign = np.sign(expr_1)
        f_1_der_log = np.log(expr_1 * f_1_der_sign) + (1.0 + self.alpha_rld) * expr_1
        digamma_expr_2_A = scipy.special.digamma(0.5 * (nu_2_vec))
        expr_2_A_sign = np.sign(digamma_expr_2_A)
        expr_2_A = (
                scipy.special.gammaln(0.5 * (nu_2_vec)) +
                np.log(expr_2_A_sign * digamma_expr_2_A) +
                np.log(0.5 * (nu_1_vec + p)) -
                scipy.special.gammaln(0.5 * (nu_2_vec + p))
        )
        digamma_expr_2_B = scipy.special.digamma(0.5 * (nu_2_vec + p))
        expr_2_B_sign = np.sign(digamma_expr_2_B)
        expr_2_B = (
                -scipy.special.gammaln(0.5 * (p + nu_2_vec)) +
                np.log(expr_2_B_sign * digamma_expr_2_B) +
                np.log(0.5 * (nu_1_vec + p)) +
                scipy.special.gammaln(0.5 * (nu_2_vec))
        )
        expr_2_B_sign = -expr_2_B_sign
        f_2_der_log, f_2_der_sign = scipy.special.logsumexp(
            a=np.array([
                expr_2_A, expr_2_B
            ]),
            b=np.array([
                expr_2_A_sign, expr_2_B_sign
            ]),
            return_sign=True,
            axis=0
        )
        expr_3 = -(0.5 * p * (np.log(nu_1_vec) + np.log(np.pi)) + self.predictive_variance_log_det)
        f_3_der_sign = np.sign(expr_3)
        f_3_der_log = f_3_log + np.log(f_3_der_sign * expr_3)
        f_1_full_expr = f_1_der_log + f_2_log + f_3_log
        f_2_full_expr = f_2_der_log + f_1_log + f_3_log
        f_3_full_expr = f_3_der_log + f_1_log + f_2_log
        log_integral_derivatives_val, log_integral_derivatives_sign = (
            scipy.special.logsumexp(
                a=np.array([
                    f_1_full_expr,
                    f_2_full_expr,
                    f_3_full_expr]),
                b=np.array([f_1_der_sign,
                            f_2_der_sign,
                            f_3_der_sign]),
                return_sign=True,
                axis=0
            ))
        integrals = self.get_log_integrals_power_divergence()
        predictive_log_probs = np.insert(
            self.one_step_ahead_predictive_log_probs.copy(), 0,
            self.r0_log_prob)
        log_constant = (1.0 / self.alpha_rld) * np.power(np.exp(predictive_log_probs), self.alpha_rld) - (
                    1.0 / (1.0 + self.alpha_rld)) * np.exp(integrals)
        expr_1_A = -2.0 * np.log(self.alpha_rld) + self.alpha_rld * predictive_log_probs
        sign_1_A = -np.ones(run_length_num)
        sign_1_B = np.sign(predictive_log_probs)
        expr_1_B = (-np.log(self.alpha_rld) +
                    np.log(sign_1_B * predictive_log_probs) +
                    self.alpha_rld * predictive_log_probs)
        expr_1_val, expr_1_sign = scipy.special.logsumexp(
            a=np.array([
                expr_1_A,
                expr_1_B
            ]),
            b=np.array([
                sign_1_A,
                sign_1_B
            ]),
            return_sign=True,
            axis=0
        )
        expr_2_A = -2.0 * np.log(self.alpha_rld + 1.0) + integrals
        sign_2_A = np.ones(run_length_num)
        sign_2_B = (-1) * log_integral_derivatives_sign
        expr_2_B = -np.log(self.alpha_rld + 1) + log_integral_derivatives_val
        expr_2_val, expr_2_sign = scipy.special.logsumexp(
            a=np.array([
                expr_2_A,
                expr_2_B
            ]),
            b=np.array([
                sign_2_A,
                sign_2_B
            ]),
            return_sign=True,
            axis=0
        )
        expr_val, expr_sign = scipy.special.logsumexp(
            a=np.array([
                expr_1_val,
                expr_2_val
            ]),
            b=np.array([
                expr_1_sign,
                expr_2_sign
            ]),
            return_sign=True,
            axis=0
        )
        final_expr = log_constant + expr_val
        final_expr_sign = expr_sign
        return final_expr, final_expr_sign

    def get_hyperparameters(self):
        return [self.a, self.b]

    def turner_hyperparameter_optimization(self, step_size):
        sign, gradient = scipy.special.logsumexp(
            a=self.model_specific_joint_log_probabilities_derivative,
            b=self.model_specific_joint_log_probabilities_derivative_sign,
            return_sign=True, axis=1)
        gradient = np.exp(gradient) * sign
        dif_old_a = self.a - self.a_old
        dif_old_b = self.b - self.b_old
        dif_old_val = np.array([dif_old_a, dif_old_b])
        dif_old_grad = gradient - self.gradient_old
        if True:
            D1_sign = np.dot(np.sign(dif_old_val), np.sign(dif_old_grad))
            D1 = min(max(np.abs(np.dot(dif_old_val, dif_old_grad)), pow(10, 5)), pow(10, -5)) * D1_sign
            D2 = min(max(np.dot(dif_old_grad, dif_old_grad), pow(10, 5)), pow(10, -5))
            alpha_1 = D1 / D2
            step_size = alpha_1
        self.a_old, self.b_old = self.a, self.b
        self.gradient_old = gradient
        self.a = min(max(self.a + gradient[0] * step_size, pow(10, -20)), pow(10, 15))
        self.b = min(max(self.b + gradient[1] * step_size, pow(10, -20)), pow(10, 15))
        a_dif, b_dif = self.a - self.a_old, self.b - self.b_old
        return [a_dif, b_dif]

    @staticmethod
    def objective_optimization(x, *args):
        (S1, S2, y, X, retained_run_lengths, run_length_distro, beta_rt,
         log_det_1_rt, log_det_2_rt, C_t_inv, b0_D_inv_b0, YY_rt,
         beta_XX_beta_rt) = args
        a, b = x
        y = y.flatten()
        a_vec = a + (retained_run_lengths + 1.0) * 0.5
        b_vec = (b + 0.5 * (b0_D_inv_b0 + YY_rt - beta_XX_beta_rt))
        log_dets = ((S1 * S2) * (np.log(b_vec) - np.log(a_vec)) + log_det_1_rt - log_det_2_rt)
        run_length_num = retained_run_lengths.shape[0]
        MVSt_log_densities = np.array([BVARNIG.mvt_log_density(
            y_flat=(np.matmul(X, beta_rt[r, :]) - y.flatten()),
            prec=(a_vec[r] / b_vec[r]) * C_t_inv[r, :, :],
            log_det=log_dets[r], prior=False, alerts=False)
            for r in range(0, run_length_num)])
        evaluation_objective = scipy.special.logsumexp(MVSt_log_densities + run_length_distro)
        return evaluation_objective

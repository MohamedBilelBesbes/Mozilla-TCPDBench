import numpy as np
from scipy import special
from .probability_model import ProbabilityModel
from .BVAR_NIG import BVARNIG


class BARNIG(ProbabilityModel):
    def __init__(self, prior_a_list, prior_b_list, lags_list, S1, S2, prior_mean_beta_list=None, prior_var_beta_list=None, prior_mean_scale_list=None, prior_var_scale_list=None, non_spd_alerts_list = None):
        if not S1*S2 == len(lags_list):
            print("ERROR! S1, S2 do not match dimension of lags_list")
            return
        else:
            self.S1, self.S2 = S1, S2
            self.lags_list = lags_list
        if prior_mean_beta_list is None and prior_mean_scale_list is None:
            self.prior_mean_beta_list = []
            for lag in lags_list:
                self.prior_mean_beta_list.append(np.zeros(lag+1))
        elif not prior_mean_beta_list is None:
            self.prior_mean_beta_list = prior_mean_beta_list
        elif not prior_mean_scale_list is None:
            self.prior_mean_beta_list = []
            for (lag, location)  in zip(lags_list, range(0, self.S1*self.S2)):
                self.prior_mean_beta_list.append(
                        np.ones(lag+1)*prior_mean_scale_list[location])
        if prior_var_beta_list is None and prior_var_scale_list is None:
            self.prior_var_beta_list = []
            for lag in lags_list:
                self.prior_var_beta_list.append(np.identity(lag+1)*100)
        elif not prior_var_beta_list is None:
            self.prior_var_beta_list = prior_var_beta_list
        elif not prior_var_scale_list is None:
            self.prior_var_beta_list = []
            for (lag, location)  in zip(lags_list, range(0, self.S1*self.S2)):
                self.prior_var_beta_list.append(
                        np.identity(lag+1)*prior_var_scale_list[location])
        self.non_spd_alerts_list = [False]*self.S1*self.S2
        self.nbh_sequence_list = []
        self.res_sequence_list = []
        for lag in lags_list:
            self.nbh_sequence_list.append([0]*lag)
            self.res_sequence_list.append([0]*lag)
        self.location_models = []
        for location in range(0, self.S1*self.S2):
            self.location_models.append(
                BVARNIG(prior_a_list[location], prior_b_list[location],
                        S1=1, S2=1, 
                        prior_mean_beta = self.prior_mean_beta_list[location],
                        prior_var_beta = self.prior_var_beta_list[location],
                        nbh_sequence=self.nbh_sequence_list[location],
                        restriction_sequence=self.res_sequence_list[location]
                        )
                )
        self.retained_run_lengths = np.array([0,0])
        self.joint_log_probabilities = 1
        self.has_lags = True
        self.lag_length = max(self.lags_list)
        self.auto_prior_update = False
        self.exo_bool = False
        self.model_log_evidence = -np.inf
        self.nbh_sequence = -1
        
    def initialization(self, X_endo, X_exo, Y_2, X_exo_2, cp_model, model_prior, padding_columns_computeXX = None, padding_column_get_x_new = None):
        print("Initializing BAR object")
        Y2 = Y_2.flatten()
        for (lag, location) in zip(self.lags_list, range(0, self.S1*self.S2)):
            X_endo_loc = X_endo[(self.lag_length - lag):,location]
            n = np.size(X_endo_loc)
            X_endo_loc = X_endo_loc.reshape(n,1)
            self.location_models[location].initialization(X_endo=X_endo_loc,
                X_exo=None, Y_2=Y2[location], X_exo_2=None,
                cp_model=cp_model, model_prior=1,
                padding_columns_computeXX=None, 
                padding_column_get_x_new=None)
        self.model_log_evidence = model_prior + np.sum([loc_mod.model_log_evidence for loc_mod in self.location_models])
        if cp_model.pmf_0(1) == 0:
            epsilon = 0.000000000001
        else:   
            epsilon = 0
        r_equal_0 = (self.model_log_evidence + 
                     np.log(cp_model.pmf_0(0) + epsilon)) 
        r_larger_0 = (self.model_log_evidence + 
                     np.log(cp_model.pmf_0(1)+ epsilon))   
        self.joint_log_probabilities = np.array([r_equal_0, r_larger_0]) 

    def evaluate_predictive_log_distribution(self, y, t):
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = np.zeros(shape=run_length_num)
        for location in range(0, self.S1*self.S2):
            log_densities += (self.location_models[location].evaluate_predictive_log_distribution(y[location], t))
        return log_densities

    def evaluate_log_prior_predictive(self, y, t):
        prior_prob = 0
        for location in range(0, self.S1*self.S2):
            prior_prob += (self.location_models[location].evaluate_log_prior_predictive(y[location],t))
        return prior_prob

    def save_NLL_fixed_pars(self, y,t):
        y = y.flatten()
        helper = np.zeros(self.retained_run_lengths.shape[0])
        for location in range(0,self.S1*self.S2):
            self.location_models[location].save_NLL_fixed_pars(y[location],t)
            helper += self.location_models[location].one_step_ahead_predictive_log_probs_fixed_pars
        self.one_step_ahead_predictive_log_probs_fixed_pars = helper

    def update_predictive_distributions(self, y_t, y_tm1, x_exo_t, x_exo_tp1, t, padding_column_tm1 = None, padding_column_t = None, r_evaluations = None):
        y_t = y_t.flatten()
        y_tm1 = y_tm1.flatten()
        for location in range(0, self.S1*self.S2):
            self.location_models[location].update_predictive_distributions(
                y_t[location], y_tm1[location], x_exo_t, x_exo_tp1, t, 
                padding_column_tm1 = None, padding_column_t = None, 
                r_evaluations = None)
        self.retained_run_lengths =  self.retained_run_lengths + 1 
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)

    def trimmer(self, kept_run_lengths):
        self.joint_log_probabilities = self.joint_log_probabilities[kept_run_lengths]
        self.retained_run_lengths = self.retained_run_lengths[kept_run_lengths]
        self.model_log_evidence = special.logsumexp(self.joint_log_probabilities)
        for location in range(0, self.S1*self.S2):
            self.location_models[location].trimmer(kept_run_lengths, BAR_submodel=True)
            
    def get_posterior_expectation(self, t, r_list=None):
        num_retained_run_lengths = np.size(self.retained_run_lengths)
        post_mean = np.zeros((num_retained_run_lengths, self.S1*self.S2))
        for location in range(0, self.S1*self.S2):
            post_mean[:,location] = (self.location_models[location].
                get_posterior_expectation(t).reshape(num_retained_run_lengths))
        return post_mean

    def get_posterior_variance(self, t, r_list=None):
        num_retained_run_lengths = np.size(self.retained_run_lengths)
        post_var = np.zeros((num_retained_run_lengths, self.S1*self.S2, self.S1*self.S2))
        for location in range(0, self.S1*self.S2):
            post_var[:,location, location] = self.location_models[location].get_posterior_variance(t).reshape(num_retained_run_lengths)
        return post_var


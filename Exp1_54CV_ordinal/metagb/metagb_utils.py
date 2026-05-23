import sys, math, numpy as np, pandas as pd, xarray as xr
import random
import datetime
# import lightgbm as lgb


from libmetagb import ffi, lib

class MetaGB:
    def __init__(self,
                 learning_rate,
                 num_trees,
                 feature_fraction,
                 bagging_fraction,
                 min_data_in_leaf,
                 max_depth,
                 max_bin,
                 pred_feature_fraction=None,
                 depth_sep=None,
                 use_random_depth_sep=0,
                 leaf_use_all_pred_feature=None,
                 leaf_only_use_bias=0,
                 cond_prior_features=[],
                 cond_prior_feature_fractions=None,
                 cond_shallow_use_prior_only=1,
                 pred_deep_use_prior_only=1,
                 cond_pred_share=0,
                 use_delta=1,
                 score_agg=0,
                 min_gain_to_split=-1e10,
                 lambda_l2 = 0,
                 normalize = 0,
                 split_sample=0,
                version=1,
                debug=0,
                seed=0):
        self.learning_rate = learning_rate
        self.num_trees = num_trees
        self.feature_fraction = feature_fraction
        if pred_feature_fraction is None:
            self.pred_feature_fraction = feature_fraction
        else:
            self.pred_feature_fraction = pred_feature_fraction
        self.depth_sep = depth_sep
        self.use_random_depth_sep = use_random_depth_sep
        if leaf_use_all_pred_feature is None:
            # Default to True for version 3, keep None for other versions
            self.leaf_use_all_pred_feature = 1 if version == 3 else leaf_use_all_pred_feature
        else:
            self.leaf_use_all_pred_feature = leaf_use_all_pred_feature
        self.leaf_only_use_bias = leaf_only_use_bias
        self.cond_shallow_use_prior_only = cond_shallow_use_prior_only
        self.cond_prior_features = cond_prior_features
        self.cond_prior_feature_fractions = cond_prior_feature_fractions
        self.pred_deep_use_prior_only = pred_deep_use_prior_only
        self.cond_pred_share = cond_pred_share
        self.use_delta = use_delta
        self.bagging_fraction = bagging_fraction
        self.min_data_in_leaf = min_data_in_leaf
        self.max_depth = max_depth
        self.max_bin = max_bin
        self.min_gain_to_split = min_gain_to_split
        self.lambda_l2 = lambda_l2
        self.normalize = normalize
        self.split_sample = split_sample
        self.score_agg = score_agg
        self.version = version
        self.debug = debug
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.model = np.full(
            [self.num_trees, 5*(2**(self.max_depth + 1) - 1)], math.nan)

    def convert_data_to_bin(self, x):
        self.bins = np.quantile(
            x, q=[i/float(self.max_bin) for i in range(1, self.max_bin+1)], axis=0)
        return np.column_stack(
            [np.digitize(x[:, i], self.bins[:, i], right=True)
             for i in range(x.shape[1])])

    def fit(self, x, y):
        if self.version == 0:
            self.fit_original(x, y)
        elif self.version == 1:
            self.fit_custom(x, y)
        elif self.version == 2:
            self.fit_memory(x, y)
        elif self.version == 3:
            self.fit_simple(x, y)

    def fit_original(self, x, y):
        n,d = x.shape[0], x.shape[1]

        x = x.astype("double")
        x_digit = self.convert_data_to_bin(x)
        x_digit = x_digit.astype("int8")
        bins = self.bins.astype("double")
        y = y.astype("double")
        tree = self.model.astype("double")

        x_ptr = ffi.cast("unsigned char*", ffi.from_buffer(x_digit))
        x_origin_ptr = ffi.cast("double *", ffi.from_buffer(x))
        bin_ptr = ffi.cast("double *", ffi.from_buffer(bins))
        y_ptr = ffi.cast("double *", ffi.from_buffer(y))
        tree_ptr = ffi.cast("double *", ffi.from_buffer(tree))

        lib.train(x_ptr,
                  x_origin_ptr,
                  bin_ptr,
                  y_ptr,
                  tree_ptr,
                  n,
                  d,
                  self.num_trees,
                  self.max_depth,
                  self.max_bin,
                  self.min_data_in_leaf,
                  self.bagging_fraction,
                  self.feature_fraction,
                  self.learning_rate)
        self.model = np.copy(tree)


    def fit_custom(self, x, y):
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        n,d = x.shape[0], x.shape[1]
        prior_cond_feature_idx = np.array(self.cond_prior_features, dtype=np.int32)
        if self.cond_prior_feature_fractions is None:
            prior_cond_feature_fractions = np.ones_like(prior_cond_feature_idx, dtype=np.float64)
        else:
            prior_cond_feature_fractions = np.array(self.cond_prior_feature_fractions, dtype=np.float64)
        num_prior_cond_features = prior_cond_feature_idx.shape[0]

        x = x.astype("double")
        x_digit = self.convert_data_to_bin(x)
        x_digit = x_digit.astype("int8")
        bins = self.bins.astype("double")
        y = y.astype("double")
        tree = self.model.astype("double")

        x_ptr = ffi.cast("unsigned char*", ffi.from_buffer(x_digit))
        x_origin_ptr = ffi.cast("double *", ffi.from_buffer(x))
        bin_ptr = ffi.cast("double *", ffi.from_buffer(bins))
        y_ptr = ffi.cast("double *", ffi.from_buffer(y))
        tree_ptr = ffi.cast("double *", ffi.from_buffer(tree))
        prior_cond_feature_idx_ptr = ffi.cast("int *", ffi.from_buffer(prior_cond_feature_idx))
        prior_cond_feature_fractions_ptr = ffi.cast("double *", ffi.from_buffer(prior_cond_feature_fractions))

        lib.train_custom(x_ptr,
                        x_origin_ptr,
                        bin_ptr,
                        y_ptr,
                        tree_ptr,
                        prior_cond_feature_idx_ptr,
                        prior_cond_feature_fractions_ptr,
                        num_prior_cond_features,
                        self.depth_sep,
                        self.use_random_depth_sep,
                        self.cond_shallow_use_prior_only,
                        self.pred_deep_use_prior_only,
                        n,
                        d,
                        d-1,
                        self.num_trees,
                        self.max_depth,
                        self.max_bin,
                        self.min_data_in_leaf,
                        self.bagging_fraction,
                        self.feature_fraction,
                        self.pred_feature_fraction,
                        self.learning_rate,
                        self.leaf_use_all_pred_feature,
                        self.cond_pred_share,
                        self.use_delta,
                        self.score_agg,
                        self.min_gain_to_split,
                        self.lambda_l2,
                        self.normalize,
                        self.split_sample,
                        self.debug
                        )
        self.model = np.copy(tree)

    def fit_memory(self, x, y):
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        n,d = x.shape[0], x.shape[1]
        prior_cond_feature_idx = np.array(self.cond_prior_features, dtype=np.int32)
        if self.cond_prior_feature_fractions is None:
            prior_cond_feature_fractions = np.ones_like(prior_cond_feature_idx, dtype=np.float64)
        else:
            prior_cond_feature_fractions = np.array(self.cond_prior_feature_fractions, dtype=np.float64)
        num_prior_cond_features = prior_cond_feature_idx.shape[0]

        x = x.astype("double")
        x_digit = self.convert_data_to_bin(x)
        x_digit = x_digit.astype("int8")
        bins = self.bins.astype("double")
        y = y.astype("double")
        tree = self.model.astype("double")

        x_ptr = ffi.cast("unsigned char*", ffi.from_buffer(x_digit))
        x_origin_ptr = ffi.cast("double *", ffi.from_buffer(x))
        bin_ptr = ffi.cast("double *", ffi.from_buffer(bins))
        y_ptr = ffi.cast("double *", ffi.from_buffer(y))
        tree_ptr = ffi.cast("double *", ffi.from_buffer(tree))
        prior_cond_feature_idx_ptr = ffi.cast("int *", ffi.from_buffer(prior_cond_feature_idx))
        prior_cond_feature_fractions_ptr = ffi.cast("double *", ffi.from_buffer(prior_cond_feature_fractions))

        lib.train_memory(x_ptr,
                        x_origin_ptr,
                        bin_ptr,
                        y_ptr,
                        tree_ptr,
                        prior_cond_feature_idx_ptr,
                        prior_cond_feature_fractions_ptr,
                        num_prior_cond_features,
                        self.depth_sep,
                        self.use_random_depth_sep,
                        self.cond_shallow_use_prior_only,
                        self.pred_deep_use_prior_only,
                        n,
                        d,
                        d-1,
                        self.num_trees,
                        self.max_depth,
                        self.max_bin,
                        self.min_data_in_leaf,
                        self.bagging_fraction,
                        self.feature_fraction,
                        self.pred_feature_fraction,
                        self.learning_rate,
                        self.leaf_use_all_pred_feature,
                        self.cond_pred_share,
                        self.use_delta,
                        self.score_agg,
                        self.min_gain_to_split,
                        self.lambda_l2,
                        self.normalize,
                        self.split_sample,
                        self.debug
                        )
        self.model = np.copy(tree)

    def fit_simple(self, x, y):
        """Version 3: Simplified training without depth_sep and cond_prior concepts."""
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        n, d = x.shape[0], x.shape[1]

        x = x.astype("double")
        x_digit = self.convert_data_to_bin(x)
        x_digit = x_digit.astype("int8")
        bins = self.bins.astype("double")
        y = y.astype("double")
        tree = self.model.astype("double")

        x_ptr = ffi.cast("unsigned char*", ffi.from_buffer(x_digit))
        x_origin_ptr = ffi.cast("double *", ffi.from_buffer(x))
        bin_ptr = ffi.cast("double *", ffi.from_buffer(bins))
        y_ptr = ffi.cast("double *", ffi.from_buffer(y))
        tree_ptr = ffi.cast("double *", ffi.from_buffer(tree))

        lib.train_simple(x_ptr,
                        x_origin_ptr,
                        bin_ptr,
                        y_ptr,
                        tree_ptr,
                        n,
                        d,
                        d-1,
                        self.num_trees,
                        self.max_depth,
                        self.max_bin,
                        self.min_data_in_leaf,
                        self.bagging_fraction,
                        self.feature_fraction,
                        self.pred_feature_fraction,
                        self.learning_rate,
                        self.leaf_use_all_pred_feature,
                        self.leaf_only_use_bias,
                        self.cond_pred_share,
                        self.use_delta,
                        self.score_agg,
                        self.min_gain_to_split,
                        self.lambda_l2,
                        self.normalize,
                        self.split_sample,
                        self.debug
                        )
        self.model = np.copy(tree)

    def fit_xreorder(self, x, y):
        n,d = x.shape[0], x.shape[1]

        x = x.astype("double")
        x_digit = self.convert_data_to_bin(x)
        x_digit = x_digit.astype("int8").transpose(1, 0).copy(order="C")
        bins = self.bins.astype("double")

        y = y.astype("double")
        tree = self.model.astype("double")

        x_ptr = ffi.cast("unsigned char*", ffi.from_buffer(x_digit))
        x_origin_ptr = ffi.cast("double *", ffi.from_buffer(x))
        bin_ptr = ffi.cast("double *", ffi.from_buffer(bins))
        y_ptr = ffi.cast("double *", ffi.from_buffer(y))
        tree_ptr = ffi.cast("double *", ffi.from_buffer(tree))

        lib.train_xreorder(x_ptr,
                        x_origin_ptr,
                        bin_ptr,
                        y_ptr,
                        tree_ptr,
                        n,
                        d,
                        self.num_trees,
                        self.max_depth,
                        self.max_bin,
                        self.min_data_in_leaf,
                        self.bagging_fraction,
                        self.feature_fraction,
                        self.learning_rate)
        self.model = np.copy(tree)

    def predict(self, x, start_tree_idx=0, end_tree_idx=-1):
        if self.version == 0:
            return self.predict_original(
                x, start_tree_idx, end_tree_idx)
        elif self.version == 1 or self.version == 2 or self.version == 3:
            return self.predict_custom(
                x, start_tree_idx, end_tree_idx)

    def predict_original(self, x, start_tree_idx=0, end_tree_idx=-1):
        tree = self.model
        tree = tree.astype("double").copy(order="C")
        x = x.astype("double")
        yhat = np.zeros(x.shape[0]).astype("double")

        tree_ptr = ffi.cast("double *", ffi.from_buffer(tree))
        x_ptr = ffi.cast("double *", ffi.from_buffer(x))
        yhat_ptr = ffi.cast("double *", ffi.from_buffer(yhat))

        lib.predict(
            x_ptr,
            tree_ptr,
            yhat_ptr,
            x.shape[0],
            x.shape[1],
            tree.shape[0],
            tree.shape[1]//5,
            self.learning_rate,
            0,
            -1
            )
        return yhat

    def predict_custom(self, x, start_tree_idx=0, end_tree_idx=-1):
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        tree = self.model
        tree = tree.astype("double").copy(order="C")
        x = x.astype("double")
        yhat = np.zeros(x.shape[0]).astype("double")

        tree_ptr = ffi.cast("double *", ffi.from_buffer(tree))
        x_ptr = ffi.cast("double *", ffi.from_buffer(x))
        yhat_ptr = ffi.cast("double *", ffi.from_buffer(yhat))

        lib.predict_custom(
            x_ptr,
            tree_ptr,
            yhat_ptr,
            x.shape[0],
            x.shape[1],
            tree.shape[0],
            tree.shape[1]//5,
            self.learning_rate,
            0,
            -1
            )
        return yhat
    
    def save_model(self, path):
        model_xarray = xr.DataArray(
            np.zeros((1, 1,
                    self.model.shape[0],
                    self.model.shape[1]),
                    dtype=np.float32),
            coords=[["0"], ["0"],
                    list(map(str, range(self.model.shape[0]))),
                    list(map(str, range(self.model.shape[1])))],
            dims=["D", "I", "S", "V"])
        model_xarray[0, 0, :, :] = self.model

        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError
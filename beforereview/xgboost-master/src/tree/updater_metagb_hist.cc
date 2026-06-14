/**
 * Copyright 2026, XGBoost Contributors
 * \file updater_metagb_hist.cc
 * \brief CPU histogram updater for MetaGB v3.
 */
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/threading_utils.h"
#include "../data/gradient_index.h"
#include "common_row_partitioner.h"
#include "dmlc/registry.h"
#include "hist/expand_entry.h"
#include "hist/hist_param.h"
#include "param.h"
#include "sample_position.h"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/gradient.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

namespace xgboost::tree {

DMLC_REGISTRY_FILE_TAG(updater_metagb_hist);

constexpr double kMetaGBEps = 1e-8;

BatchParam MetaGBHistBatch(TrainParam const* param) {
  return {param->max_bin, param->sparse_threshold};
}

struct MetaGBStats {
  double grad_x{0.0};
  double hess_xx{0.0};
  double hess{0.0};

  void Add(double grad, double hess, double x) {
    grad_x += grad * x;
    hess_xx += hess * x * x;
    this->hess += hess;
  }

  MetaGBStats operator-(MetaGBStats const& rhs) const {
    MetaGBStats out;
    out.grad_x = grad_x - rhs.grad_x;
    out.hess_xx = hess_xx - rhs.hess_xx;
    out.hess = hess - rhs.hess;
    return out;
  }
};

struct MetaGBSplitInfo {
  bst_bin_t split_bin{0};
};

struct MetaGBBinSplit {
  double gain{-std::numeric_limits<double>::infinity()};
  MetaGBStats left;
  MetaGBStats right;
  bool valid{false};
};

struct MetaGBNodeRows {
  std::vector<bst_node_t> cond;
  std::vector<bst_node_t> pred;
};

enum class MetaGBHistBuildMode {
  kAuto,
  kRowSerial,
  kCfParallel,
  kRowParallel,
};

struct MetaGBMultiStats {
  std::vector<double> grad_x;
  std::vector<double> hess_xx;
  std::vector<double> hess;

  explicit MetaGBMultiStats(std::size_t n_targets = 0)
      : grad_x(n_targets, 0.0), hess_xx(n_targets, 0.0), hess(n_targets, 0.0) {}

  void Resize(std::size_t n_targets) {
    grad_x.assign(n_targets, 0.0);
    hess_xx.assign(n_targets, 0.0);
    hess.assign(n_targets, 0.0);
  }

  void Add(std::size_t t, double grad, double hess_value, double x) {
    grad_x[t] += grad * x;
    hess_xx[t] += hess_value * x * x;
    hess[t] += hess_value;
  }
};

struct MetaGBMultiSplitInfo {
  bst_feature_t pred_feature{std::numeric_limits<bst_feature_t>::max()};
  std::vector<bst_feature_t> pred_features;
  bst_bin_t split_bin{0};
  std::vector<float> left_weight;
  std::vector<float> right_weight;
  double left_hess{0.0};
  double right_hess{0.0};
};

double Score(MetaGBStats const& stats, double lambda) {
  return stats.grad_x * stats.grad_x / (stats.hess_xx + lambda + kMetaGBEps);
}

float Weight(MetaGBStats const& stats, double lambda) {
  return static_cast<float>(-stats.grad_x / (stats.hess_xx + lambda + kMetaGBEps));
}

double Score(MetaGBMultiStats const& stats, double lambda) {
  double score{0.0};
  for (std::size_t t = 0; t < stats.grad_x.size(); ++t) {
    score += stats.grad_x[t] * stats.grad_x[t] / (stats.hess_xx[t] + lambda + kMetaGBEps);
  }
  return score;
}

std::vector<float> Weight(MetaGBMultiStats const& stats, double lambda, float learning_rate) {
  std::vector<float> weight(stats.grad_x.size(), 0.0f);
  for (std::size_t t = 0; t < stats.grad_x.size(); ++t) {
    weight[t] = static_cast<float>(
        -stats.grad_x[t] / (stats.hess_xx[t] + lambda + kMetaGBEps) * learning_rate);
  }
  return weight;
}

double Median(std::vector<double>* values) {
  auto& v = *values;
  CHECK(!v.empty());
  auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
}

double OriginalRandUniform() {
  return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}

std::size_t OriginalRandIndex(std::size_t upper) {
  CHECK_GT(upper, 0UL);
  return static_cast<std::size_t>(std::rand()) % upper;
}

float DenseValue(GHistIndexMatrix const& gmat, common::HistogramCuts const& cuts, std::size_t ridx,
                 bst_feature_t feature, bst_feature_t n_features) {
  if (feature == n_features) {
    return 1.0f;
  }
  auto bin = gmat.GetGindex(ridx, feature);
  if (bin == static_cast<bst_bin_t>(-1)) {
    return 0.0f;
  }
  // Original MetaGB v3 uses bins[x_digit, feature], i.e. the upper cut value of
  // the current digitized bin, not the lower-bound representative used by XGBoost prediction.
  return cuts.Values()[bin];
}

float SparsePageValue(SparsePage const& page, std::size_t ridx, bst_feature_t feature,
                      bst_feature_t n_features) {
  if (feature == n_features) {
    return 1.0f;
  }
  auto inst = page.GetView()[ridx];
  for (auto const& entry : inst) {
    if (entry.index == feature) {
      return std::isnan(entry.fvalue) ? 0.0f : entry.fvalue;
    }
    if (entry.index > feature) {
      break;
    }
  }
  return 0.0f;
}

bst_bin_t FeatureBin(GHistIndexMatrix const& gmat, std::size_t ridx, bst_feature_t feature) {
  return gmat.GetGindex(ridx, feature);
}

std::vector<bst_feature_t> SampleFeatures(bst_feature_t n_features, float fraction,
                                          bool require_non_empty) {
  std::vector<bst_feature_t> features;
  features.reserve(n_features);
  for (bst_feature_t f = 0; f < n_features; ++f) {
    if (OriginalRandUniform() <= static_cast<double>(fraction)) {
      features.push_back(f);
    }
  }
  if (require_non_empty && features.empty() && n_features > 0) {
    features.push_back(static_cast<bst_feature_t>(OriginalRandIndex(n_features)));
  }
  return features;
}

std::vector<bst_feature_t> PredFeatures(bst_feature_t n_features, TrainParam const& param,
                                        std::vector<bst_feature_t> const& cond_features) {
  std::vector<bst_feature_t> pred;
  if (param.metagb_cond_pred_share) {
    pred = cond_features;
  } else {
    pred = SampleFeatures(n_features, param.metagb_pred_colsample_bytree, false);
  }
  pred.push_back(n_features);
  return pred;
}

class MetaGBHistMaker : public TreeUpdater {
  common::Monitor monitor_;
  HistMakerTrainParam hist_param_;
  TrainParam const* param_{nullptr};
  std::vector<CommonRowPartitioner> partitioner_;
  DMatrix* p_last_fmat_{nullptr};
  RegTree const* p_last_tree_{nullptr};
  std::unordered_map<bst_node_t, MetaGBSplitInfo> split_info_;
  std::unordered_map<bst_node_t, MetaGBNodeRows> node_rows_;
  std::vector<bst_node_t> last_node_position_;
  std::vector<float> raw_values_;
  std::vector<bst_bin_t> digitized_bins_;
  std::vector<double> bins_;
  std::vector<double> pred_train_value_cache_;
  std::vector<double> leaf_train_value_cache_;
  std::vector<MetaGBStats> leaf_workspace_;
  std::vector<MetaGBStats> hist_workspace_;
  std::vector<MetaGBMultiStats> multi_leaf_workspace_;
  std::vector<MetaGBMultiStats> multi_hist_workspace_;
  std::vector<float> multi_leaf_weights_;
  std::vector<float> cond_grad_workspace_;
  std::vector<float> cond_hess_workspace_;
  std::vector<MetaGBStats> hist_3d_workspace_;  // 3D histogram: [n_cond][n_pred][n_bins]
  std::vector<MetaGBBinSplit> split_workspace_;
  std::vector<MetaGBStats> hist_chunk_workspace_;
  // Per-thread workspaces for parallel cf evaluation
  std::vector<std::vector<MetaGBStats>> thread_hist_chunk_;
  std::vector<std::vector<MetaGBBinSplit>> thread_split_;
  // Histogram subtraction cache: node -> 3D hist [n_cf][n_pred][n_bins]
  std::unordered_map<bst_node_t, std::vector<MetaGBStats>> node_hist_cache_;
  std::unordered_map<bst_node_t, MetaGBMultiSplitInfo> multi_split_info_;
  DMatrix const* p_cached_bins_fmat_{nullptr};
  std::size_t n_rows_{0};
  bst_feature_t n_features_{0};
  int max_bin_{0};
  bool rand_seeded_{false};
  MetaGBHistBuildMode hist_build_mode_{MetaGBHistBuildMode::kAuto};

  void ConfigureHistBuildModeFromEnv() {
    auto const* env = std::getenv("METAGB_HIST_BUILD_MODE");
    if (env == nullptr || std::strlen(env) == 0 || std::strcmp(env, "auto") == 0) {
      hist_build_mode_ = MetaGBHistBuildMode::kAuto;
      return;
    }
    if (std::strcmp(env, "row_serial") == 0) {
      hist_build_mode_ = MetaGBHistBuildMode::kRowSerial;
      return;
    }
    if (std::strcmp(env, "cf_parallel") == 0) {
      hist_build_mode_ = MetaGBHistBuildMode::kCfParallel;
      return;
    }
    if (std::strcmp(env, "row_parallel") == 0) {
      hist_build_mode_ = MetaGBHistBuildMode::kRowParallel;
      return;
    }
    LOG(WARNING) << "Unknown METAGB_HIST_BUILD_MODE='" << env
                 << "', falling back to row_serial.";
    hist_build_mode_ = MetaGBHistBuildMode::kRowSerial;
  }

  [[nodiscard]] MetaGBHistBuildMode ResolveHistBuildMode(std::size_t n_cond,
                                                         std::size_t n_cf,
                                                         std::size_t n_pred) const {
    if (hist_build_mode_ != MetaGBHistBuildMode::kAuto) {
      return hist_build_mode_;
    }
    if (ctx_->Threads() <= 1 || n_cf <= 1) {
      return MetaGBHistBuildMode::kRowSerial;
    }
    // Small feature spaces are dominated by OpenMP overhead. Wide tables benefit
    // from disjoint cf-slice writes even though rows are scanned per cf.
    auto work_items = n_cond * n_cf * n_pred;
    if (n_cf >= 4 && work_items >= 50000) {
      return MetaGBHistBuildMode::kCfParallel;
    }
    return MetaGBHistBuildMode::kRowSerial;
  }

  [[nodiscard]] bool HasSplit(bst_node_t nidx) const {
    return split_info_.find(nidx) != split_info_.cend();
  }

  [[nodiscard]] MetaGBNodeRows const& NodeRows(bst_node_t nidx) const {
    auto it = node_rows_.find(nidx);
    CHECK(it != node_rows_.cend()) << "Missing MetaGB rows for node: " << nidx;
    return it->second;
  }

  void EnsureOriginalRandSeeded() {
    if (!rand_seeded_) {
      std::srand(1);
      rand_seeded_ = true;
    }
  }

  [[nodiscard]] MetaGBNodeRows SampleActiveRows(std::size_t n_rows, float subsample,
                                                int split_sample) const {
    MetaGBNodeRows rows;
    rows.cond.reserve(n_rows);
    rows.pred.reserve(n_rows);
    std::size_t n_active{0};
    for (std::size_t ridx = 0; ridx < n_rows; ++ridx) {
      if (OriginalRandUniform() <= static_cast<double>(subsample)) {
        auto row = static_cast<bst_node_t>(ridx);
        if (split_sample == 1) {
          if (OriginalRandUniform() <= 0.5) {
            rows.cond.push_back(row);
          } else {
            rows.pred.push_back(row);
          }
        } else if (split_sample == 2) {
          if (ridx <= n_rows / 2) {
            rows.cond.push_back(row);
          } else {
            rows.pred.push_back(row);
          }
        } else {
          rows.cond.push_back(row);
          rows.pred.push_back(row);
        }
        ++n_active;
      }
    }
    if (n_active == 0 && n_rows > 0) {
      rows.cond.push_back(static_cast<bst_node_t>(OriginalRandIndex(n_rows)));
      rows.pred.push_back(static_cast<bst_node_t>(OriginalRandIndex(n_rows)));
    }
    return rows;
  }

  [[nodiscard]] std::size_t BinOffset(bst_bin_t bin, bst_feature_t feature) const {
    return static_cast<std::size_t>(bin) * static_cast<std::size_t>(n_features_ + 1) + feature;
  }

  [[nodiscard]] float RawValue(std::size_t ridx, bst_feature_t feature) const {
    CHECK_LT(ridx, n_rows_);
    CHECK_LT(feature, n_features_);
    return raw_values_[ridx * n_features_ + feature];
  }

  // Hot-path inline accessor - no bounds check for performance
  [[nodiscard]] bst_bin_t OriginalBinUnchecked(std::size_t ridx, bst_feature_t feature) const {
    return digitized_bins_[ridx * n_features_ + feature];
  }

  [[nodiscard]] bst_bin_t OriginalBin(std::size_t ridx, bst_feature_t feature) const {
    CHECK_LT(feature, n_features_);
    return digitized_bins_[ridx * n_features_ + feature];
  }

  [[nodiscard]] double TrainValue(std::size_t ridx, bst_feature_t feature) const {
    if (feature == n_features_) {
      return 1.0;
    }
    auto bin = OriginalBin(ridx, feature);
    return bins_[BinOffset(bin, feature)];
  }

  [[nodiscard]] double SplitValue(bst_bin_t bin, bst_feature_t feature) const {
    return bins_[BinOffset(bin, feature)];
  }

  // OPTIMIZED: Row-major layout for better cache locality
  // Layout: cache[ridx * n_pred + pred_idx] - consecutive access in inner loop
  std::size_t cached_n_pred_{0};  // Store n_pred for pred_train_value_cache_
  std::size_t cached_n_leaf_pred_{0};  // Store n_pred for leaf_train_value_cache_

  [[nodiscard]] std::size_t PredValueOffset(std::size_t pred_idx, std::size_t ridx,
                                            std::size_t n_pred) const {
    return ridx * n_pred + pred_idx;
  }

  void InitTrainValueCache(std::vector<bst_feature_t> const& pred_features,
                           std::vector<double>* cache) const {
    auto n_pred = pred_features.size();
    cache->resize(n_pred * n_rows_);
    for (std::size_t ridx = 0; ridx < n_rows_; ++ridx) {
      for (std::size_t pi = 0; pi < n_pred; ++pi) {
        (*cache)[ridx * n_pred + pi] = TrainValue(ridx, pred_features[pi]);
      }
    }
  }

  [[nodiscard]] bool OriginalBinsCacheValid(DMatrix const* p_fmat, std::size_t n_rows,
                                            bst_feature_t n_features, int max_bin) const {
    return p_cached_bins_fmat_ == p_fmat && n_rows_ == n_rows && n_features_ == n_features &&
           max_bin_ == max_bin && !bins_.empty() && !digitized_bins_.empty();
  }

  void InitOriginalBins(DMatrix const* p_fmat, SparsePage const& page, std::size_t n_rows,
                        bst_feature_t n_features, int max_bin) {
    if (OriginalBinsCacheValid(p_fmat, n_rows, n_features, max_bin)) {
      return;
    }
    p_cached_bins_fmat_ = p_fmat;
    n_rows_ = n_rows;
    n_features_ = n_features;
    max_bin_ = max_bin;
    raw_values_.assign(n_rows_ * n_features_, 0.0f);
    for (std::size_t ridx = 0; ridx < n_rows_; ++ridx) {
      auto inst = page.GetView()[ridx];
      for (auto const& entry : inst) {
        if (entry.index < n_features_) {
          raw_values_[ridx * n_features_ + entry.index] =
              std::isnan(entry.fvalue) ? 0.0f : entry.fvalue;
        }
      }
    }

    bins_.assign(static_cast<std::size_t>(max_bin_) * static_cast<std::size_t>(n_features_ + 1),
                 1.0);
    digitized_bins_.assign(n_rows_ * n_features_, 0);
    std::vector<double> values(n_rows_);
    for (bst_feature_t f = 0; f < n_features_; ++f) {
      for (std::size_t ridx = 0; ridx < n_rows_; ++ridx) {
        values[ridx] = RawValue(ridx, f);
      }
      std::sort(values.begin(), values.end());
      for (bst_bin_t b = 0; b < max_bin_; ++b) {
        double q = static_cast<double>(b + 1) / static_cast<double>(max_bin_);
        double pos = static_cast<double>(n_rows_ - 1) * q;
        auto lo = static_cast<std::size_t>(std::floor(pos));
        auto hi = static_cast<std::size_t>(std::ceil(pos));
        double frac = pos - static_cast<double>(lo);
        bins_[BinOffset(b, f)] = values[lo] * (1.0 - frac) + values[hi] * frac;
      }
      for (std::size_t ridx = 0; ridx < n_rows_; ++ridx) {
        auto value = static_cast<double>(RawValue(ridx, f));
        bst_bin_t bin = 0;
        while (bin + 1 < max_bin_ && value > bins_[BinOffset(bin, f)]) {
          ++bin;
        }
        digitized_bins_[ridx * n_features_ + f] = bin;
      }
    }
  }

  [[nodiscard]] std::size_t HistOffset(std::size_t pred_idx, bst_bin_t bin) const {
    return pred_idx * static_cast<std::size_t>(max_bin_) + static_cast<std::size_t>(bin);
  }

  [[nodiscard]] int MinDataInLeaf() const {
    if (param_->metagb_min_data_in_leaf >= 0) {
      return param_->metagb_min_data_in_leaf;
    }
    return static_cast<int>(std::max(0.0f, param_->min_child_weight));
  }

  MetaGBStats LeafStats(linalg::MatrixView<GradientPair const> gpair,
                        common::Span<bst_node_t const> rows, bst_feature_t pred_feature,
                        bst_feature_t n_features) const {
    MetaGBStats stats;
    for (auto ridx : rows) {
      auto gp = gpair(ridx, 0);
      if (gp.GetHess() <= 0.0f) {
        continue;
      }
      auto x = TrainValue(ridx, pred_feature);
      stats.Add(gp.GetGrad(), gp.GetHess(), x);
    }
    return stats;
  }

  void BestLeaf(linalg::MatrixView<GradientPair const> gpair,
                common::Span<bst_node_t const> rows,
                std::vector<bst_feature_t> const& pred_features, bst_feature_t n_features,
                std::vector<double> const& train_value_cache,
                bst_feature_t* pred_feature, float* weight) {
    double best_score = -std::numeric_limits<double>::infinity();
    *pred_feature = n_features;
    *weight = 0.0f;
    auto n_pred = pred_features.size();
    if (leaf_workspace_.size() < n_pred) {
      leaf_workspace_.resize(n_pred);
    }
    std::fill(leaf_workspace_.begin(), leaf_workspace_.begin() + n_pred, MetaGBStats{});

    auto leaf_work = rows.size() * n_pred;
    if (ctx_->Threads() > 1 && n_pred >= 8 && leaf_work >= 200000) {
#pragma omp parallel for schedule(dynamic) num_threads(ctx_->Threads())
      for (std::size_t pi = 0; pi < n_pred; ++pi) {
        MetaGBStats stats;
        for (auto ridx : rows) {
          auto gp = gpair(ridx, 0);
          if (gp.GetHess() <= 0.0f) {
            continue;
          }
          auto x = train_value_cache[PredValueOffset(pi, ridx, n_pred)];
          stats.Add(gp.GetGrad(), gp.GetHess(), x);
        }
        leaf_workspace_[pi] = stats;
      }
    } else {
      for (auto ridx : rows) {
        auto gp = gpair(ridx, 0);
        if (gp.GetHess() <= 0.0f) {
          continue;
        }
        auto grad = gp.GetGrad();
        auto hess = gp.GetHess();
        for (std::size_t pi = 0; pi < n_pred; ++pi) {
          auto x = train_value_cache[PredValueOffset(pi, ridx, n_pred)];
          leaf_workspace_[pi].Add(grad, hess, x);
        }
      }
    }

    for (std::size_t pi = 0; pi < n_pred; ++pi) {
      auto pf = pred_features[pi];
      auto const& stats = leaf_workspace_[pi];
      auto score = Score(stats, param_->metagb_lambda_l2);
      if (score > best_score || (score == best_score && pf < *pred_feature)) {
        best_score = score;
        *pred_feature = pf;
        *weight = Weight(stats, param_->metagb_lambda_l2);
      }
    }
  }

  CPUExpandEntry FindBestSplit(linalg::MatrixView<GradientPair const> gpair,
                               RegTree* p_tree,
                               bst_node_t nidx, bst_node_t depth,
                               std::vector<bst_feature_t> const& cond_features,
                               std::vector<bst_feature_t> const& split_pred_features,
                               std::vector<bst_feature_t> const& leaf_pred_features) {
    auto const& rows = NodeRows(nidx);
    CPUExpandEntry best{nidx, depth};

    bst_feature_t best_leaf_feature;
    float best_leaf_weight;
    auto const& leaf_cache =
        param_->metagb_leaf_only_use_bias ? leaf_train_value_cache_ : pred_train_value_cache_;
    BestLeaf(gpair, rows.pred, leaf_pred_features, p_tree->NumFeatures(), leaf_cache,
             &best_leaf_feature, &best_leaf_weight);
    (*p_tree)[nidx].SetLeaf(param_->learning_rate * best_leaf_weight);
    p_tree->SetMetaGBLeaf(nidx, best_leaf_feature);

    if (rows.pred.size() <= static_cast<std::size_t>(MinDataInLeaf()) || rows.cond.empty()) {
      return best;
    }
    if (param_->max_depth > 0 && depth >= param_->max_depth) {
      return best;
    }

    double best_gain = -std::numeric_limits<double>::infinity();
    MetaGBSplitInfo best_meta;
    GradStats best_left_sum;
    GradStats best_right_sum;

    if (ctx_->Threads() <= 1 && !param_->metagb_hist_subtract) {
      auto n_pred = split_pred_features.size();
      bst_bin_t n_bins = static_cast<bst_bin_t>(max_bin_);
      auto hist_size = n_pred * static_cast<std::size_t>(n_bins);
      if (hist_workspace_.size() < hist_size) {
        hist_workspace_.resize(hist_size);
      }
      if (split_workspace_.size() < hist_size) {
        split_workspace_.resize(hist_size);
      }

      for (auto cf : cond_features) {
        if (n_bins <= 1) {
          continue;
        }
        std::fill(hist_workspace_.begin(), hist_workspace_.begin() + hist_size, MetaGBStats{});
        for (auto ridx : rows.cond) {
          auto gp = gpair(ridx, 0);
          if (gp.GetHess() <= 0.0f) {
            continue;
          }
          auto local_bin = OriginalBinUnchecked(ridx, cf);
          if (local_bin < 0 || local_bin >= n_bins) {
            continue;
          }
          auto grad = gp.GetGrad();
          auto hess = gp.GetHess();
          for (std::size_t pi = 0; pi < n_pred; ++pi) {
            auto x = pred_train_value_cache_[PredValueOffset(pi, ridx, cached_n_pred_)];
            hist_workspace_[HistOffset(pi, local_bin)].Add(grad, hess, x);
          }
        }

        std::fill(split_workspace_.begin(), split_workspace_.begin() + hist_size, MetaGBBinSplit{});
        for (std::size_t pi = 0; pi < n_pred; ++pi) {
          MetaGBStats total;
          for (bst_bin_t b = 0; b < n_bins; ++b) {
            auto const& s = hist_workspace_[HistOffset(pi, b)];
            total.grad_x += s.grad_x;
            total.hess_xx += s.hess_xx;
            total.hess += s.hess;
          }
          auto base_score = Score(total, param_->metagb_lambda_l2);
          MetaGBStats left;
          for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
            auto const& s = hist_workspace_[HistOffset(pi, b)];
            left.grad_x += s.grad_x;
            left.hess_xx += s.hess_xx;
            left.hess += s.hess;
            auto right = total - left;
            if (left.hess <= kMetaGBEps || right.hess <= kMetaGBEps) {
              continue;
            }
            auto gain = Score(left, param_->metagb_lambda_l2) +
                        Score(right, param_->metagb_lambda_l2) - base_score;
            split_workspace_[HistOffset(pi, b)] = MetaGBBinSplit{gain, left, right, true};
            if (param_->metagb_score_agg == 0 && gain > best_gain) {
              best_gain = gain;
              best.split.Update(static_cast<float>(gain), cf,
                                static_cast<float>(SplitValue(b, cf)), false, false,
                                GradStats{0.0, left.hess}, GradStats{0.0, right.hess});
              best_left_sum = GradStats{0.0, left.hess};
              best_right_sum = GradStats{0.0, right.hess};
              best_meta.split_bin = b;
            }
          }
        }

        if (param_->metagb_score_agg == 0) {
          continue;
        }
        for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
          if (!split_workspace_[HistOffset(0, b)].valid) {
            continue;
          }
          double gain = -std::numeric_limits<double>::infinity();
          if (param_->metagb_score_agg == 1) {
            gain = 0.0;
            for (std::size_t pi = 0; pi < n_pred; ++pi) {
              gain += split_workspace_[HistOffset(pi, b)].gain / static_cast<double>(n_pred);
            }
          } else if (param_->metagb_score_agg == 2) {
            std::vector<double> scores;
            scores.reserve(n_pred);
            for (std::size_t pi = 0; pi < n_pred; ++pi) {
              scores.push_back(split_workspace_[HistOffset(pi, b)].gain);
            }
            gain = Median(&scores);
          } else if (param_->metagb_score_agg == 3) {
            gain = split_workspace_[HistOffset(OriginalRandIndex(n_pred), b)].gain;
          }
          if (gain > best_gain) {
            auto const& sw = split_workspace_[HistOffset(0, b)];
            best_gain = gain;
            best.split.Update(static_cast<float>(gain), cf,
                              static_cast<float>(SplitValue(b, cf)), false, false,
                              GradStats{0.0, sw.left.hess}, GradStats{0.0, sw.right.hess});
            best_left_sum = GradStats{0.0, sw.left.hess};
            best_right_sum = GradStats{0.0, sw.right.hess};
            best_meta.split_bin = b;
          }
        }
      }

      if (best_gain == -std::numeric_limits<double>::infinity() ||
          best_gain < param_->metagb_min_gain_to_split) {
        return CPUExpandEntry{nidx, depth};
      }
      std::size_t pred_left_count{0};
      std::size_t pred_right_count{0};
      for (auto ridx : rows.pred) {
        if (OriginalBinUnchecked(ridx, best.split.SplitIndex()) <= best_meta.split_bin) {
          ++pred_left_count;
        } else {
          ++pred_right_count;
        }
      }
      if (pred_left_count == 0 || pred_right_count == 0) {
        return CPUExpandEntry{nidx, depth};
      }
      best.split.left_sum = best_left_sum;
      best.split.right_sum = best_right_sum;
      split_info_[nidx] = best_meta;
      return best;
    }

    // 方案 C: 预取 rows.cond 的 grad/hess，避免 cf 循环内重复随机访问
    std::size_t n_cond = rows.cond.size();
    cond_grad_workspace_.resize(n_cond);
    cond_hess_workspace_.resize(n_cond);
    for (std::size_t i = 0; i < n_cond; ++i) {
      auto gp = gpair(rows.cond[i], 0);
      cond_grad_workspace_[i] = gp.GetGrad();
      cond_hess_workspace_[i] = gp.GetHess();
    }

    bst_bin_t n_bins = static_cast<bst_bin_t>(max_bin_);
    auto n_cf = cond_features.size();
    auto n_pred = split_pred_features.size();
    auto hist_size = n_pred * static_cast<std::size_t>(n_bins);
    auto hist3d_size = n_cf * hist_size;

    hist_3d_workspace_.assign(hist3d_size, MetaGBStats{});
    auto& node_hist = hist_3d_workspace_;
    bool built_via_subtraction = false;

    if (param_->metagb_hist_subtract && nidx != RegTree::kRoot) {
      auto parent_nid = (*p_tree)[nidx].Parent();
      auto parent_it = node_hist_cache_.find(parent_nid);
      if (parent_it != node_hist_cache_.end() &&
          parent_it->second.size() == hist3d_size) {
        auto sibling = ((*p_tree)[parent_nid].LeftChild() == nidx)
                           ? (*p_tree)[parent_nid].RightChild()
                           : (*p_tree)[parent_nid].LeftChild();
        auto sib_it = node_hist_cache_.find(sibling);
        if (sib_it != node_hist_cache_.end() &&
            sib_it->second.size() == hist3d_size) {
          // this_node = parent - sibling  (element-wise, valid for all [cf][pi][bin])
          auto const& par = parent_it->second;
          auto const& sib = sib_it->second;
          common::ParallelFor(hist3d_size, ctx_->Threads(), [&](std::size_t k) {
            node_hist[k] = par[k] - sib[k];
          });
          built_via_subtraction = true;
          node_hist_cache_.erase(parent_it);  // parent no longer needed after both children processed
        }
      }
    }

    auto resolved_hist_build_mode = ResolveHistBuildMode(n_cond, n_cf, n_pred);

    if (!built_via_subtraction &&
        resolved_hist_build_mode == MetaGBHistBuildMode::kRowParallel &&
        ctx_->Threads() > 1 && n_cond > 1) {
      int n_threads = ctx_->Threads();
      thread_hist_chunk_.resize(n_threads);
      for (int tid = 0; tid < n_threads; ++tid) {
        thread_hist_chunk_[tid].assign(hist3d_size, MetaGBStats{});
      }

#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (std::size_t i = 0; i < n_cond; ++i) {
        if (cond_hess_workspace_[i] <= 0.0f) continue;
        int tid = omp_get_thread_num();
        auto& local_hist = thread_hist_chunk_[tid];
        auto ridx = rows.cond[i];
        auto grad = cond_grad_workspace_[i];
        auto hess = cond_hess_workspace_[i];
        for (std::size_t cf_idx = 0; cf_idx < n_cf; ++cf_idx) {
          auto bin = OriginalBinUnchecked(ridx, cond_features[cf_idx]);
          if (bin < 0 || bin >= n_bins) continue;
          auto* cf_slice = local_hist.data() + cf_idx * hist_size;
          for (std::size_t pi = 0; pi < n_pred; ++pi) {
            auto x = pred_train_value_cache_[PredValueOffset(pi, ridx, cached_n_pred_)];
            cf_slice[HistOffset(pi, bin)].Add(grad, hess, x);
          }
        }
      }

#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (std::size_t k = 0; k < hist3d_size; ++k) {
        MetaGBStats sum;
        for (int tid = 0; tid < n_threads; ++tid) {
          auto const& local = thread_hist_chunk_[tid][k];
          sum.grad_x += local.grad_x;
          sum.hess_xx += local.hess_xx;
          sum.hess += local.hess;
        }
        node_hist[k] = sum;
      }
    } else if (!built_via_subtraction &&
               resolved_hist_build_mode == MetaGBHistBuildMode::kCfParallel &&
               ctx_->Threads() > 1 && n_cf > 1) {
#pragma omp parallel for schedule(dynamic) num_threads(ctx_->Threads())
      for (std::size_t cf_idx = 0; cf_idx < n_cf; ++cf_idx) {
        auto cf = cond_features[cf_idx];
        auto* cf_slice = node_hist.data() + cf_idx * hist_size;
        for (std::size_t i = 0; i < n_cond; ++i) {
          if (cond_hess_workspace_[i] <= 0.0f) continue;
          auto ridx = rows.cond[i];
          auto bin = OriginalBinUnchecked(ridx, cf);
          if (bin < 0 || bin >= n_bins) continue;
          auto grad = cond_grad_workspace_[i];
          auto hess = cond_hess_workspace_[i];
          auto const* row_pred_values =
              pred_train_value_cache_.data() + ridx * cached_n_pred_;
          for (std::size_t pi = 0; pi < n_pred; ++pi) {
            auto x = row_pred_values[pi];
            cf_slice[HistOffset(pi, bin)].Add(grad, hess, x);
          }
        }
      }
    } else if (!built_via_subtraction) {
      for (std::size_t i = 0; i < n_cond; ++i) {
        if (cond_hess_workspace_[i] <= 0.0f) continue;
        auto ridx = rows.cond[i];
          auto grad = cond_grad_workspace_[i];
          auto hess = cond_hess_workspace_[i];
          auto const* row_pred_values =
              pred_train_value_cache_.data() + ridx * cached_n_pred_;
          for (std::size_t cf_idx = 0; cf_idx < n_cf; ++cf_idx) {
            auto bin = OriginalBinUnchecked(ridx, cond_features[cf_idx]);
            if (bin < 0 || bin >= n_bins) continue;
            auto* cf_slice = node_hist.data() + cf_idx * hist_size;
            for (std::size_t pi = 0; pi < n_pred; ++pi) {
              auto x = row_pred_values[pi];
              cf_slice[HistOffset(pi, bin)].Add(grad, hess, x);
            }
          }
      }
    }

    // Parallelize split evaluation over cf_idx. node_hist is read-only here.
    int n_threads = ctx_->Threads();
    thread_split_.resize(n_threads);

    std::vector<double> thr_best_gain(n_threads, -std::numeric_limits<double>::infinity());
    std::vector<MetaGBSplitInfo> thr_best_meta(n_threads);
    std::vector<GradStats> thr_best_left(n_threads);
    std::vector<GradStats> thr_best_right(n_threads);
    std::vector<CPUExpandEntry> thr_best_entry(n_threads, CPUExpandEntry{nidx, depth});

#pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (std::size_t cf_idx = 0; cf_idx < n_cf; ++cf_idx) {
      if (n_bins <= 1) continue;
      int tid = omp_get_thread_num();
      auto cf = cond_features[cf_idx];
      auto const* cf_hist = node_hist.data() + cf_idx * hist_size;

      double local_best_gain = -std::numeric_limits<double>::infinity();
      MetaGBSplitInfo local_meta;
      GradStats local_left_sum, local_right_sum;
      CPUExpandEntry local_best{nidx, depth};

      if (param_->metagb_score_agg == 0) {
        for (std::size_t pi = 0; pi < n_pred; ++pi) {
          MetaGBStats total;
          for (bst_bin_t b = 0; b < n_bins; ++b) {
            auto const& s = cf_hist[HistOffset(pi, b)];
            total.grad_x += s.grad_x;
            total.hess_xx += s.hess_xx;
            total.hess += s.hess;
          }
          auto base_score = Score(total, param_->metagb_lambda_l2);
          MetaGBStats left;
          for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
            auto const& s = cf_hist[HistOffset(pi, b)];
            left.grad_x += s.grad_x;
            left.hess_xx += s.hess_xx;
            left.hess += s.hess;
            auto right = total - left;
            if (left.hess <= kMetaGBEps || right.hess <= kMetaGBEps) continue;
            auto gain = Score(left, param_->metagb_lambda_l2) +
                        Score(right, param_->metagb_lambda_l2) - base_score;
            if (gain > local_best_gain) {
              local_best_gain = gain;
              local_best.split.Update(static_cast<float>(gain), cf,
                                      static_cast<float>(SplitValue(b, cf)), false, false,
                                      GradStats{0.0, left.hess},
                                      GradStats{0.0, right.hess});
              local_left_sum = GradStats{0.0, left.hess};
              local_right_sum = GradStats{0.0, right.hess};
              local_meta.split_bin = b;
            }
          }
        }
      } else {
        auto& local_split = thread_split_[tid];
        if (local_split.size() < hist_size) local_split.resize(hist_size);
        std::fill(local_split.begin(), local_split.begin() + hist_size, MetaGBBinSplit{});

        for (std::size_t pi = 0; pi < n_pred; ++pi) {
          MetaGBStats total;
          for (bst_bin_t b = 0; b < n_bins; ++b) {
            auto const& s = cf_hist[HistOffset(pi, b)];
            total.grad_x += s.grad_x;
            total.hess_xx += s.hess_xx;
            total.hess += s.hess;
          }
          auto base_score = Score(total, param_->metagb_lambda_l2);
          MetaGBStats left;
          for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
            auto const& s = cf_hist[HistOffset(pi, b)];
            left.grad_x += s.grad_x;
            left.hess_xx += s.hess_xx;
            left.hess += s.hess;
            auto right = total - left;
            if (left.hess <= kMetaGBEps || right.hess <= kMetaGBEps) continue;
            auto gain = Score(left, param_->metagb_lambda_l2) +
                        Score(right, param_->metagb_lambda_l2) - base_score;
            local_split[HistOffset(pi, b)] = MetaGBBinSplit{gain, left, right, true};
          }
        }

        for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
          if (!local_split[HistOffset(0, b)].valid) continue;
          double gain = -std::numeric_limits<double>::infinity();
          if (param_->metagb_score_agg == 1) {
            gain = 0.0;
            for (std::size_t pi = 0; pi < n_pred; ++pi)
              gain += local_split[HistOffset(pi, b)].gain / static_cast<double>(n_pred);
          } else if (param_->metagb_score_agg == 2) {
            std::vector<double> scores;
            scores.reserve(n_pred);
            for (std::size_t pi = 0; pi < n_pred; ++pi)
              scores.push_back(local_split[HistOffset(pi, b)].gain);
            gain = Median(&scores);
          } else if (param_->metagb_score_agg == 3) {
            gain = local_split[HistOffset(OriginalRandIndex(n_pred), b)].gain;
          }
          if (gain > local_best_gain) {
            auto const& sw = local_split[HistOffset(0, b)];
            local_best_gain = gain;
            local_best.split.Update(static_cast<float>(gain), cf,
                                    static_cast<float>(SplitValue(b, cf)), false, false,
                                    GradStats{0.0, sw.left.hess},
                                    GradStats{0.0, sw.right.hess});
            local_left_sum = GradStats{0.0, sw.left.hess};
            local_right_sum = GradStats{0.0, sw.right.hess};
            local_meta.split_bin = b;
          }
        }
      }

      if (local_best_gain > thr_best_gain[tid]) {
        thr_best_gain[tid] = local_best_gain;
        thr_best_meta[tid] = local_meta;
        thr_best_left[tid] = local_left_sum;
        thr_best_right[tid] = local_right_sum;
        thr_best_entry[tid] = local_best;
      }
    }

    for (int t = 0; t < n_threads; ++t) {
      if (thr_best_gain[t] > best_gain) {
        best_gain = thr_best_gain[t];
        best = thr_best_entry[t];
        best_meta = thr_best_meta[t];
        best_left_sum = thr_best_left[t];
        best_right_sum = thr_best_right[t];
      }
    }

    if (best_gain == -std::numeric_limits<double>::infinity() ||
        best_gain < param_->metagb_min_gain_to_split) {
      return CPUExpandEntry{nidx, depth};
    }
    std::size_t pred_left_count{0};
    std::size_t pred_right_count{0};
    for (auto ridx : rows.pred) {
      if (OriginalBinUnchecked(ridx, best.split.SplitIndex()) <= best_meta.split_bin) {
        ++pred_left_count;
      } else {
        ++pred_right_count;
      }
    }
    if (pred_left_count == 0 || pred_right_count == 0) {
      return CPUExpandEntry{nidx, depth};
    }
    best.split.left_sum = best_left_sum;
    best.split.right_sum = best_right_sum;
    if (param_->metagb_hist_subtract) {
      node_hist_cache_[nidx] = node_hist;
    }
    split_info_[nidx] = best_meta;
    return best;
  }

  void BestMultiLeaf(linalg::MatrixView<GradientPair const> gpair,
                     common::Span<bst_node_t const> rows,
                     std::vector<bst_feature_t> const& pred_features, bst_feature_t n_features,
                     std::vector<double> const& train_value_cache,
                     std::vector<bst_feature_t>* out_pred_features,
                     std::vector<float>* weights) {
    auto n_targets = gpair.Shape(1);
    auto n_pred = pred_features.size();
    if (multi_leaf_workspace_.size() < n_pred) {
      multi_leaf_workspace_.resize(n_pred);
    }
    for (std::size_t pi = 0; pi < n_pred; ++pi) {
      multi_leaf_workspace_[pi].Resize(n_targets);
    }

    for (auto ridx : rows) {
      for (std::size_t t = 0; t < n_targets; ++t) {
        auto gp = gpair(ridx, t);
        if (gp.GetHess() <= 0.0f) {
          continue;
        }
        for (std::size_t pi = 0; pi < n_pred; ++pi) {
          auto x = train_value_cache[PredValueOffset(pi, ridx, n_pred)];
          multi_leaf_workspace_[pi].Add(t, gp.GetGrad(), gp.GetHess(), x);
        }
      }
    }

    std::vector<double> best_score(n_targets, -std::numeric_limits<double>::infinity());
    out_pred_features->assign(n_targets, n_features);
    weights->assign(n_targets, 0.0f);
    for (std::size_t pi = 0; pi < n_pred; ++pi) {
      auto pf = pred_features[pi];
      for (std::size_t t = 0; t < n_targets; ++t) {
        auto const& stats = multi_leaf_workspace_[pi];
        auto score = stats.grad_x[t] * stats.grad_x[t] /
                     (stats.hess_xx[t] + param_->metagb_lambda_l2 + kMetaGBEps);
        if (score > best_score[t] || (score == best_score[t] && pf < (*out_pred_features)[t])) {
          best_score[t] = score;
          (*out_pred_features)[t] = pf;
          (*weights)[t] = static_cast<float>(
              -stats.grad_x[t] / (stats.hess_xx[t] + param_->metagb_lambda_l2 + kMetaGBEps) *
              param_->learning_rate * param_->metagb_vector_leaf_scale);
        }
      }
    }
  }

  double SumMultiHess(linalg::MatrixView<GradientPair const> gpair,
                      common::Span<bst_node_t const> rows) const {
    double sum_hess{0.0};
    auto n_targets = gpair.Shape(1);
    for (auto ridx : rows) {
      for (std::size_t t = 0; t < n_targets; ++t) {
        auto hess = gpair(ridx, t).GetHess();
        if (hess > 0.0f) {
          sum_hess += hess;
        }
      }
    }
    return sum_hess;
  }

  void InitMultiRoot(linalg::MatrixView<GradientPair const> gpair, RegTree* p_tree,
                     std::vector<bst_feature_t> const& leaf_pred_features) {
    std::vector<bst_feature_t> pred_features;
    std::vector<float> weights;
    auto const& leaf_cache =
        param_->metagb_leaf_only_use_bias ? leaf_train_value_cache_ : pred_train_value_cache_;
    auto const& rows = NodeRows(RegTree::kRoot).pred;
    BestMultiLeaf(gpair, rows, leaf_pred_features, p_tree->NumFeatures(), leaf_cache,
                  &pred_features, &weights);
    for (bst_target_t t = 0; t < pred_features.size(); ++t) {
      p_tree->SetMetaGBLeaf(RegTree::kRoot, t, pred_features[t]);
    }
    p_tree->SetMetaGBLeaf(RegTree::kRoot, pred_features.front());
    auto root_weight = linalg::MakeVec(weights.data(), weights.size(), DeviceOrd::CPU());
    p_tree->SetRoot(root_weight, static_cast<float>(SumMultiHess(gpair, rows)));
  }

  CPUExpandEntry FindBestMultiSplit(linalg::MatrixView<GradientPair const> gpair, RegTree* p_tree,
                                    bst_node_t nidx, bst_node_t depth,
                                    std::vector<bst_feature_t> const& cond_features,
                                    std::vector<bst_feature_t> const& split_pred_features,
                                    std::vector<bst_feature_t> const& leaf_pred_features) {
    auto const& rows = NodeRows(nidx);
    CPUExpandEntry best{nidx, depth};

    std::vector<bst_feature_t> best_leaf_features;
    std::vector<float> best_leaf_weight;
    auto const& leaf_cache =
        param_->metagb_leaf_only_use_bias ? leaf_train_value_cache_ : pred_train_value_cache_;
    BestMultiLeaf(gpair, rows.pred, leaf_pred_features, p_tree->NumFeatures(), leaf_cache,
                  &best_leaf_features, &best_leaf_weight);
    for (bst_target_t t = 0; t < best_leaf_features.size(); ++t) {
      p_tree->SetMetaGBLeaf(nidx, t, best_leaf_features[t]);
    }
    p_tree->SetMetaGBLeaf(nidx, best_leaf_features.front());

    if (rows.pred.size() <= static_cast<std::size_t>(MinDataInLeaf()) || rows.cond.empty()) {
      return best;
    }
    if (param_->max_depth > 0 && depth >= param_->max_depth) {
      return best;
    }

    bst_bin_t n_bins = static_cast<bst_bin_t>(max_bin_);
    if (n_bins <= 1) {
      return best;
    }

    auto n_targets = gpair.Shape(1);
    auto n_pred = split_pred_features.size();
    auto hist_size = n_pred * static_cast<std::size_t>(n_bins);
    if (multi_hist_workspace_.size() < hist_size) {
      multi_hist_workspace_.resize(hist_size);
    }

    double best_gain = -std::numeric_limits<double>::infinity();
    MetaGBMultiSplitInfo best_meta;

    for (auto cf : cond_features) {
      for (std::size_t i = 0; i < hist_size; ++i) {
        multi_hist_workspace_[i].Resize(n_targets);
      }
      for (auto ridx : rows.cond) {
        auto local_bin = OriginalBinUnchecked(ridx, cf);
        if (local_bin < 0 || local_bin >= n_bins) {
          continue;
        }
        for (std::size_t t = 0; t < n_targets; ++t) {
          auto gp = gpair(ridx, t);
          if (gp.GetHess() <= 0.0f) {
            continue;
          }
          for (std::size_t pi = 0; pi < n_pred; ++pi) {
            auto x = pred_train_value_cache_[PredValueOffset(pi, ridx, cached_n_pred_)];
            multi_hist_workspace_[HistOffset(pi, local_bin)].Add(t, gp.GetGrad(), gp.GetHess(), x);
          }
        }
      }

      std::vector<MetaGBMultiStats> totals(n_pred, MetaGBMultiStats(n_targets));
      for (std::size_t pi = 0; pi < n_pred; ++pi) {
        for (bst_bin_t b = 0; b < n_bins; ++b) {
          auto const& s = multi_hist_workspace_[HistOffset(pi, b)];
          for (std::size_t t = 0; t < n_targets; ++t) {
            totals[pi].grad_x[t] += s.grad_x[t];
            totals[pi].hess_xx[t] += s.hess_xx[t];
            totals[pi].hess[t] += s.hess[t];
          }
        }
      }

      std::vector<MetaGBMultiStats> lefts(n_pred, MetaGBMultiStats(n_targets));
      for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
        std::vector<double> split_target_gains(n_targets, 0.0);
        double split_left_hess{0.0};
        double split_right_hess{0.0};
        bool valid{true};
        std::vector<bst_feature_t> split_pred_by_target(n_targets, p_tree->NumFeatures());
        std::vector<float> split_left_weight(n_targets, 0.0f);
        std::vector<float> split_right_weight(n_targets, 0.0f);
        for (std::size_t pi = 0; pi < n_pred; ++pi) {
          auto const& s = multi_hist_workspace_[HistOffset(pi, b)];
          for (std::size_t t = 0; t < n_targets; ++t) {
            lefts[pi].grad_x[t] += s.grad_x[t];
            lefts[pi].hess_xx[t] += s.hess_xx[t];
            lefts[pi].hess[t] += s.hess[t];
          }
        }

        for (std::size_t t = 0; t < n_targets; ++t) {
          double best_target_gain = -std::numeric_limits<double>::infinity();
          bst_feature_t best_target_feature = p_tree->NumFeatures();
          float best_left_weight{0.0f};
          float best_right_weight{0.0f};
          double best_left_hess{0.0};
          double best_right_hess{0.0};
          for (std::size_t pi = 0; pi < n_pred; ++pi) {
            auto const& left = lefts[pi];
            auto const& total = totals[pi];
            auto right_grad_x = total.grad_x[t] - left.grad_x[t];
            auto right_hess_xx = total.hess_xx[t] - left.hess_xx[t];
            auto right_hess = total.hess[t] - left.hess[t];
            if (left.hess[t] <= kMetaGBEps || right_hess <= kMetaGBEps) {
              continue;
            }
            auto base_score =
                total.grad_x[t] * total.grad_x[t] /
                (total.hess_xx[t] + param_->metagb_lambda_l2 + kMetaGBEps);
            auto left_score =
                left.grad_x[t] * left.grad_x[t] /
                (left.hess_xx[t] + param_->metagb_lambda_l2 + kMetaGBEps);
            auto right_score =
                right_grad_x * right_grad_x /
                (right_hess_xx + param_->metagb_lambda_l2 + kMetaGBEps);
            auto target_gain = left_score + right_score - base_score;
            auto pf = split_pred_features[pi];
            if (target_gain > best_target_gain ||
                (target_gain == best_target_gain && pf < best_target_feature)) {
              best_target_gain = target_gain;
              best_target_feature = pf;
              best_left_weight = static_cast<float>(
                  -left.grad_x[t] / (left.hess_xx[t] + param_->metagb_lambda_l2 + kMetaGBEps) *
                  param_->learning_rate * param_->metagb_vector_leaf_scale);
              best_right_weight = static_cast<float>(
                  -right_grad_x / (right_hess_xx + param_->metagb_lambda_l2 + kMetaGBEps) *
                  param_->learning_rate * param_->metagb_vector_leaf_scale);
              best_left_hess = left.hess[t];
              best_right_hess = right_hess;
            }
          }
          if (best_target_gain == -std::numeric_limits<double>::infinity()) {
            valid = false;
            break;
          }
          split_target_gains[t] = best_target_gain;
          split_left_hess += best_left_hess;
          split_right_hess += best_right_hess;
          split_pred_by_target[t] = best_target_feature;
          split_left_weight[t] = best_left_weight;
          split_right_weight[t] = best_right_weight;
        }

        if (!valid) {
          continue;
        }
        double split_gain{0.0};
        if (param_->metagb_vector_score_agg == 0) {
          split_gain = std::accumulate(split_target_gains.cbegin(), split_target_gains.cend(), 0.0);
        } else if (param_->metagb_vector_score_agg == 1) {
          split_gain = std::accumulate(split_target_gains.cbegin(), split_target_gains.cend(), 0.0) /
                       static_cast<double>(n_targets);
        } else if (param_->metagb_vector_score_agg == 2) {
          split_gain = *std::min_element(split_target_gains.cbegin(), split_target_gains.cend());
        } else {
          LOG(FATAL) << "MetaGB vector branch supports metagb_vector_score_agg in [0, 2].";
        }
        if (split_gain > best_gain) {
          best_gain = split_gain;
          best.split.Update(static_cast<float>(split_gain), cf,
                            static_cast<float>(SplitValue(b, cf)), false, false,
                            GradStats{0.0, split_left_hess},
                            GradStats{0.0, split_right_hess});
          best_meta.pred_feature = split_pred_by_target.front();
          best_meta.pred_features = std::move(split_pred_by_target);
          best_meta.split_bin = b;
          best_meta.left_weight = std::move(split_left_weight);
          best_meta.right_weight = std::move(split_right_weight);
          best_meta.left_hess = split_left_hess;
          best_meta.right_hess = split_right_hess;
        }
      }
    }

    if (best_gain == -std::numeric_limits<double>::infinity() ||
        best_gain < param_->metagb_min_gain_to_split) {
      return CPUExpandEntry{nidx, depth};
    }
    std::size_t pred_left_count{0};
    std::size_t pred_right_count{0};
    for (auto ridx : rows.pred) {
      if (OriginalBinUnchecked(ridx, best.split.SplitIndex()) <= best_meta.split_bin) {
        ++pred_left_count;
      } else {
        ++pred_right_count;
      }
    }
    if (pred_left_count == 0 || pred_right_count == 0) {
      return CPUExpandEntry{nidx, depth};
    }
    multi_split_info_[nidx] = std::move(best_meta);
    return best;
  }

  void ApplyTreeSplit(CPUExpandEntry const& candidate, RegTree* p_tree) {
    auto it = split_info_.find(candidate.nid);
    CHECK(it != split_info_.cend());
    auto const& meta = it->second;
    auto parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    p_tree->ExpandNode(candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
                       candidate.split.DefaultLeft(), 0.0f, 0.0f, 0.0f, candidate.split.loss_chg,
                       parent_sum.GetHess(), candidate.split.left_sum.GetHess(),
                       candidate.split.right_sum.GetHess());
    auto left = p_tree->LeftChild(candidate.nid);
    auto right = p_tree->RightChild(candidate.nid);
    // Split aggregation chooses only the conditional split. Final leaf models are
    // selected independently from each leaf's predictive rows, matching original MetaGB v3.
    p_tree->SetMetaGBLeaf(left, p_tree->NumFeatures());
    p_tree->SetMetaGBLeaf(right, p_tree->NumFeatures());

    auto const& parent_rows = NodeRows(candidate.nid);
    auto split_feature = candidate.split.SplitIndex();
    auto split_bin = meta.split_bin;
    auto& left_rows = node_rows_[left];
    auto& right_rows = node_rows_[right];
    left_rows.cond.clear();
    left_rows.pred.clear();
    right_rows.cond.clear();
    right_rows.pred.clear();
    left_rows.cond.reserve(parent_rows.cond.size());
    right_rows.cond.reserve(parent_rows.cond.size());
    left_rows.pred.reserve(parent_rows.pred.size());
    right_rows.pred.reserve(parent_rows.pred.size());
    for (auto ridx : parent_rows.cond) {
      if (OriginalBinUnchecked(ridx, split_feature) <= split_bin) {
        left_rows.cond.push_back(ridx);
      } else {
        right_rows.cond.push_back(ridx);
      }
    }
    for (auto ridx : parent_rows.pred) {
      if (OriginalBinUnchecked(ridx, split_feature) <= split_bin) {
        left_rows.pred.push_back(ridx);
      } else {
        right_rows.pred.push_back(ridx);
      }
    }
  }

  void ApplyMultiTreeSplit(CPUExpandEntry const& candidate, RegTree* p_tree) {
    auto it = multi_split_info_.find(candidate.nid);
    CHECK(it != multi_split_info_.cend());
    auto const& meta = it->second;
    auto parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    std::vector<float> base_weight_storage(meta.left_weight.size(), 0.0f);
    auto base_weight = linalg::MakeVec(base_weight_storage.data(), base_weight_storage.size(),
                                       DeviceOrd::CPU());
    auto left_weight =
        linalg::MakeVec(meta.left_weight.data(), meta.left_weight.size(), DeviceOrd::CPU());
    auto right_weight =
        linalg::MakeVec(meta.right_weight.data(), meta.right_weight.size(), DeviceOrd::CPU());
    p_tree->ExpandNode(candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
                       candidate.split.DefaultLeft(), base_weight, left_weight, right_weight,
                       candidate.split.loss_chg, parent_sum.GetHess(),
                       static_cast<float>(meta.left_hess), static_cast<float>(meta.right_hess));
    auto left = p_tree->LeftChild(candidate.nid);
    auto right = p_tree->RightChild(candidate.nid);
    p_tree->SetMetaGBLeaf(left, meta.pred_feature);
    p_tree->SetMetaGBLeaf(right, meta.pred_feature);

    auto const& parent_rows = NodeRows(candidate.nid);
    auto split_feature = candidate.split.SplitIndex();
    auto split_bin = meta.split_bin;
    auto& left_rows = node_rows_[left];
    auto& right_rows = node_rows_[right];
    left_rows.cond.clear();
    left_rows.pred.clear();
    right_rows.cond.clear();
    right_rows.pred.clear();
    left_rows.cond.reserve(parent_rows.cond.size());
    right_rows.cond.reserve(parent_rows.cond.size());
    left_rows.pred.reserve(parent_rows.pred.size());
    right_rows.pred.reserve(parent_rows.pred.size());
    for (auto ridx : parent_rows.cond) {
      if (OriginalBinUnchecked(ridx, split_feature) <= split_bin) {
        left_rows.cond.push_back(ridx);
      } else {
        right_rows.cond.push_back(ridx);
      }
    }
    for (auto ridx : parent_rows.pred) {
      if (OriginalBinUnchecked(ridx, split_feature) <= split_bin) {
        left_rows.pred.push_back(ridx);
      } else {
        right_rows.pred.push_back(ridx);
      }
    }
  }

  void RefreshMultiLeaves(linalg::MatrixView<GradientPair const> gpair, RegTree* p_tree,
                          std::vector<bst_feature_t> const& leaf_pred_features) {
    std::vector<bst_node_t> leaves;
    auto tree_view = p_tree->HostMtView();
    tree_view.WalkTree([&](bst_node_t nidx) {
      if (tree_view.IsLeaf(nidx)) {
        leaves.push_back(nidx);
      }
      return true;
    });
    auto n_targets = gpair.Shape(1);
    multi_leaf_weights_.assign(leaves.size() * n_targets, 0.0f);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
      auto nidx = leaves[i];
      std::vector<bst_feature_t> pred_features;
      std::vector<float> weights;
      auto const& leaf_cache =
          param_->metagb_leaf_only_use_bias ? leaf_train_value_cache_ : pred_train_value_cache_;
      BestMultiLeaf(gpair, NodeRows(nidx).pred, leaf_pred_features, p_tree->NumFeatures(),
                    leaf_cache, &pred_features, &weights);
      for (bst_target_t t = 0; t < pred_features.size(); ++t) {
        p_tree->SetMetaGBLeaf(nidx, t, pred_features[t]);
      }
      p_tree->SetMetaGBLeaf(nidx, pred_features.front());
      CHECK_EQ(weights.size(), n_targets);
      std::copy(weights.cbegin(), weights.cend(),
                multi_leaf_weights_.begin() + static_cast<std::ptrdiff_t>(i * n_targets));
    }
    p_tree->SetLeaves(std::move(leaves),
                      common::Span<float const>{multi_leaf_weights_.data(),
                                                multi_leaf_weights_.size()});
  }

  void RefreshLeaves(linalg::MatrixView<GradientPair const> gpair, RegTree* p_tree,
                     std::vector<bst_feature_t> const& leaf_pred_features) {
    auto const& leaf_cache =
        param_->metagb_leaf_only_use_bias ? leaf_train_value_cache_ : pred_train_value_cache_;
    auto tree_view = p_tree->HostScView();
    tree_view.WalkTree([&](bst_node_t nidx) {
      if (tree_view.IsLeaf(nidx)) {
        bst_feature_t pred_feature;
        float weight;
        BestLeaf(gpair, NodeRows(nidx).pred, leaf_pred_features, p_tree->NumFeatures(), leaf_cache,
                 &pred_feature, &weight);
        (*p_tree)[nidx].SetLeaf(param_->learning_rate * weight);
        p_tree->SetMetaGBLeaf(nidx, pred_feature);
      }
      return true;
    });
  }

  void InitData(DMatrix* p_fmat, RegTree const* p_tree) {
    monitor_.Start(__func__);
    p_last_fmat_ = p_fmat;
    std::size_t page_idx = 0;
    for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, MetaGBHistBatch(param_))) {
      if (page_idx < partitioner_.size()) {
        partitioner_[page_idx].Reset(ctx_, page.Size(), page.base_rowid, p_fmat->Info().IsColumnSplit());
      } else {
        partitioner_.emplace_back(ctx_, page.Size(), page.base_rowid, p_fmat->Info().IsColumnSplit());
      }
      ++page_idx;
    }
    partitioner_.resize(page_idx);
    p_last_tree_ = p_tree;
    monitor_.Stop(__func__);
  }

  void UpdatePosition(DMatrix* p_fmat, RegTree const* p_tree,
                      std::vector<CPUExpandEntry> const& applied) {
    std::size_t page_id{0};
    for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, MetaGBHistBatch(param_))) {
      partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree->HostScView());
      ++page_id;
    }
  }

  static bst_node_t PredictLeaf(SparsePage const& page, std::size_t ridx, RegTree const& tree) {
    bst_node_t nidx = RegTree::kRoot;
    while (tree.LeftChild(nidx) != RegTree::kInvalidNodeId) {
      bst_feature_t split_index;
      float split_cond;
      if (tree.IsMultiTarget()) {
        auto const tree_view = tree.HostMtView();
        split_index = tree_view.SplitIndex(nidx);
        split_cond = tree_view.SplitCond(nidx);
      } else {
        auto const tree_view = tree.HostScView();
        split_index = tree_view.SplitIndex(nidx);
        split_cond = tree_view.SplitCond(nidx);
      }
      auto value = SparsePageValue(page, ridx, split_index, tree.NumFeatures());
      nidx = value <= split_cond ? tree.LeftChild(nidx) : tree.RightChild(nidx);
    }
    return nidx;
  }

 public:
  explicit MetaGBHistMaker(Context const* ctx, ObjInfo const*)
      : TreeUpdater{ctx} {
    monitor_.Init(__func__);
  }

  void Configure(Args const& args) override {
    hist_param_.UpdateAllowUnknown(args);
    ConfigureHistBuildModeFromEnv();
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("hist_train_param"), &hist_param_);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["hist_train_param"] = ToJson(hist_param_);
  }

  [[nodiscard]] char const* Name() const override { return "grow_metagb_histmaker"; }

  void Update(TrainParam const* param, GradientContainer* in_gpair, DMatrix* p_fmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              std::vector<RegTree*> const& trees) override {
    CHECK(param->enable_metagb);
    CHECK_EQ(trees.size(), 1U) << "MetaGB first implementation supports one tree at a time.";
    CHECK_GE(param->metagb_score_agg, 0) << "MetaGB supports score_agg in [0, 3].";
    CHECK_LE(param->metagb_score_agg, 3) << "MetaGB supports score_agg in [0, 3].";
    this->param_ = param;
    auto* p_tree = trees.front();
    if (p_tree->IsMultiTarget()) {
      CHECK_GT(in_gpair->Grad()->Shape(1), 1U)
          << "MetaGB vector branch expects multiple gradient columns.";
      CHECK_EQ(param->metagb_score_agg, 0)
          << "MetaGB vector branch currently supports metagb_score_agg=0.";
      CHECK_GE(param->metagb_vector_score_agg, 0)
          << "MetaGB vector branch supports metagb_vector_score_agg in [0, 2].";
      CHECK_LE(param->metagb_vector_score_agg, 2)
          << "MetaGB vector branch supports metagb_vector_score_agg in [0, 2].";
      p_tree->EnableMetaGB();
      split_info_.clear();
      multi_split_info_.clear();
      node_rows_.clear();
      last_node_position_.clear();
      node_hist_cache_.clear();
      InitData(p_fmat, p_tree);

      auto h_gpair = in_gpair->Grad()->HostView();
      EnsureOriginalRandSeeded();

      auto const& page = *(p_fmat->GetBatches<SparsePage>().begin());
      CHECK_EQ(page.Size(), h_gpair.Shape(0))
          << "MetaGB initial implementation expects a single in-memory row page.";
      InitOriginalBins(p_fmat, page, page.Size(), p_tree->NumFeatures(), param->max_bin);
      auto sampled_rows =
          SampleActiveRows(page.Size(), param->subsample, param->metagb_split_sample);
      auto cond_features = SampleFeatures(p_tree->NumFeatures(), param->colsample_bytree, true);
      auto pred_features = PredFeatures(p_tree->NumFeatures(), *param, cond_features);
      std::vector<bst_feature_t> leaf_pred_features =
          param->metagb_leaf_only_use_bias ? std::vector<bst_feature_t>{p_tree->NumFeatures()}
                                           : pred_features;
      cached_n_pred_ = pred_features.size();
      InitTrainValueCache(pred_features, &pred_train_value_cache_);
      if (param->metagb_leaf_only_use_bias) {
        InitTrainValueCache(leaf_pred_features, &leaf_train_value_cache_);
      } else {
        leaf_train_value_cache_.clear();
      }
      auto& root_rows = node_rows_[RegTree::kRoot];
      root_rows = std::move(sampled_rows);
      InitMultiRoot(h_gpair, p_tree, leaf_pred_features);

      auto root = FindBestMultiSplit(h_gpair, p_tree, RegTree::kRoot, 0, cond_features,
                                     pred_features, leaf_pred_features);
      std::vector<CPUExpandEntry> expand_set;
      if (multi_split_info_.find(root.nid) != multi_split_info_.cend()) {
        expand_set.push_back(root);
      }
      while (!expand_set.empty()) {
        std::vector<CPUExpandEntry> applied;
        std::vector<CPUExpandEntry> best_splits;
        for (auto const& candidate : expand_set) {
          ApplyMultiTreeSplit(candidate, p_tree);
          applied.push_back(candidate);
        }
        for (auto const& candidate : applied) {
          auto left = p_tree->LeftChild(candidate.nid);
          auto right = p_tree->RightChild(candidate.nid);
          auto depth = candidate.depth + 1;
          auto l_best = FindBestMultiSplit(h_gpair, p_tree, left, depth, cond_features,
                                           pred_features, leaf_pred_features);
          auto r_best = FindBestMultiSplit(h_gpair, p_tree, right, depth, cond_features,
                                           pred_features, leaf_pred_features);
          if (multi_split_info_.find(l_best.nid) != multi_split_info_.cend()) {
            best_splits.push_back(l_best);
          }
          if (multi_split_info_.find(r_best.nid) != multi_split_info_.cend()) {
            best_splits.push_back(r_best);
          }
        }
        expand_set = std::move(best_splits);
      }
      RefreshMultiLeaves(h_gpair, p_tree, leaf_pred_features);

      auto& h_out_position = out_position[0].HostVector();
      h_out_position.resize(h_gpair.Shape(0));
      auto const& cache_page = *(p_fmat->GetBatches<SparsePage>().begin());
      common::ParallelFor(h_gpair.Shape(0), ctx_->Threads(), [&](std::size_t ridx) {
        h_out_position[ridx] =
            SamplePosition::Encode(PredictLeaf(cache_page, ridx, *p_tree), true);
      });
      last_node_position_ = h_out_position;
      return;
    }

    CHECK_EQ(in_gpair->Grad()->Shape(1), 1U)
        << "MetaGB expects one gradient column per scalar tree. Multi-class classification must "
           "use XGBoost's class-wise scalar-tree path unless multi_strategy=multi_output_tree is "
           "set for the experimental vector branch.";
    p_tree->EnableMetaGB();
    split_info_.clear();
    multi_split_info_.clear();
    node_rows_.clear();
    last_node_position_.clear();
    node_hist_cache_.clear();
    InitData(p_fmat, p_tree);

    auto h_gpair = in_gpair->Grad()->HostView();
    EnsureOriginalRandSeeded();

    auto const& page = *(p_fmat->GetBatches<SparsePage>().begin());
    CHECK_EQ(page.Size(), h_gpair.Shape(0))
        << "MetaGB initial implementation expects a single in-memory row page.";
    InitOriginalBins(p_fmat, page, page.Size(), p_tree->NumFeatures(), param->max_bin);
    auto sampled_rows =
        SampleActiveRows(page.Size(), param->subsample, param->metagb_split_sample);
    auto cond_features = SampleFeatures(p_tree->NumFeatures(), param->colsample_bytree, true);
    auto pred_features = PredFeatures(p_tree->NumFeatures(), *param, cond_features);
    std::vector<bst_feature_t> leaf_pred_features =
        param->metagb_leaf_only_use_bias ? std::vector<bst_feature_t>{p_tree->NumFeatures()}
                                         : pred_features;
    cached_n_pred_ = pred_features.size();  // Set before InitTrainValueCache
    InitTrainValueCache(pred_features, &pred_train_value_cache_);
    if (param->metagb_leaf_only_use_bias) {
      InitTrainValueCache(leaf_pred_features, &leaf_train_value_cache_);
    } else {
      leaf_train_value_cache_.clear();
    }
    auto& root_rows = node_rows_[RegTree::kRoot];
    root_rows = std::move(sampled_rows);

    auto root =
        FindBestSplit(h_gpair, p_tree, RegTree::kRoot, 0, cond_features, pred_features,
                      leaf_pred_features);
    std::vector<CPUExpandEntry> expand_set;
    if (HasSplit(root.nid)) {
      expand_set.push_back(root);
    }
    while (!expand_set.empty()) {
      std::vector<CPUExpandEntry> applied;
      std::vector<CPUExpandEntry> best_splits;
      for (auto const& candidate : expand_set) {
        ApplyTreeSplit(candidate, p_tree);
        applied.push_back(candidate);
      }
      for (auto const& candidate : applied) {
        auto left = p_tree->LeftChild(candidate.nid);
        auto right = p_tree->RightChild(candidate.nid);
        auto depth = candidate.depth + 1;
        auto l_best = FindBestSplit(h_gpair, p_tree, left, depth, cond_features, pred_features,
                                    leaf_pred_features);
        auto r_best = FindBestSplit(h_gpair, p_tree, right, depth, cond_features, pred_features,
                                    leaf_pred_features);
        if (HasSplit(l_best.nid)) {
          best_splits.push_back(l_best);
        }
        if (HasSplit(r_best.nid)) {
          best_splits.push_back(r_best);
        }
      }
      expand_set = std::move(best_splits);
    }
    RefreshLeaves(h_gpair, p_tree, leaf_pred_features);

    auto& h_out_position = out_position[0].HostVector();
    h_out_position.resize(h_gpair.Shape(0));
    auto const& cache_page = *(p_fmat->GetBatches<SparsePage>().begin());
    common::ParallelFor(h_gpair.Shape(0), ctx_->Threads(), [&](std::size_t ridx) {
      h_out_position[ridx] = SamplePosition::Encode(PredictLeaf(cache_page, ridx, *p_tree), true);
    });
    last_node_position_ = h_out_position;
  }

  bool UpdatePredictionCache(DMatrix const* p_fmat,
                             common::Span<HostDeviceVector<bst_node_t>> node_position,
                             linalg::MatrixView<float> out_preds) override {
    if (!p_last_fmat_ || !p_last_tree_ || p_fmat != p_last_fmat_ || node_position.size() != 1) {
      return false;
    }
    CHECK(out_preds.Device().IsCPU());
    if (p_last_tree_->IsMultiTarget()) {
      return false;
    }
    CHECK_EQ(out_preds.Shape(1), 1U);
    auto const position = node_position.front().ConstHostSpan();
    CHECK_EQ(out_preds.Shape(0), position.size());
    CHECK_EQ(position.size(), last_node_position_.size());
    auto const& tree = *p_last_tree_;
    auto const& page = *(p_last_fmat_->GetBatches<SparsePage>().begin());
    common::ParallelFor(out_preds.Shape(0), ctx_->Threads(), [&](std::size_t ridx) {
      auto nidx = SamplePosition::Decode(position[ridx]);
      auto pred_feature = tree.MetaGBLeafFeature(nidx);
      auto value = SparsePageValue(page, ridx, pred_feature, tree.NumFeatures());
      out_preds(ridx, 0) += tree.HostScView().LeafValue(nidx) * value;
    });
    return true;
  }

  [[nodiscard]] bool HasNodePosition() const override { return true; }
};

XGBOOST_REGISTER_TREE_UPDATER(MetaGBHistMaker, "grow_metagb_histmaker")
    .describe("Grow tree using MetaGB CPU histogram.")
    .set_body([](Context const* ctx, ObjInfo const* task) { return new MetaGBHistMaker{ctx, task}; });

}  // namespace xgboost::tree

/**
 * Copyright 2026, XGBoost Contributors
 * \file updater_metagb_hist.cc
 * \brief CPU histogram updater for MetaGB v3.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/threading_utils.h"
#include "../data/gradient_index.h"
#include "common_row_partitioner.h"
#include "dmlc/registry.h"
#include "hist/expand_entry.h"
#include "hist/hist_param.h"
#include "hist/sampler.h"
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
  std::size_t count{0};

  void Add(double grad, double hess, double x) {
    grad_x += grad * x;
    hess_xx += hess * x * x;
    this->hess += hess;
    ++count;
  }

  MetaGBStats operator-(MetaGBStats const& rhs) const {
    MetaGBStats out;
    out.grad_x = grad_x - rhs.grad_x;
    out.hess_xx = hess_xx - rhs.hess_xx;
    out.hess = hess - rhs.hess;
    out.count = count - rhs.count;
    return out;
  }
};

struct MetaGBSplitInfo {
  bst_feature_t pred_feature{std::numeric_limits<bst_feature_t>::max()};
  bst_bin_t split_bin{0};
  float left_weight{0.0f};
  float right_weight{0.0f};
};

struct MetaGBBinSplit {
  double gain{-std::numeric_limits<double>::infinity()};
  MetaGBStats left;
  MetaGBStats right;
  bool valid{false};
};

double Score(MetaGBStats const& stats, double lambda) {
  return stats.grad_x * stats.grad_x / (stats.hess_xx + lambda + kMetaGBEps);
}

float Weight(MetaGBStats const& stats, double lambda) {
  return static_cast<float>(-stats.grad_x / (stats.hess_xx + lambda + kMetaGBEps));
}

double Median(std::vector<double>* values) {
  auto& v = *values;
  CHECK(!v.empty());
  auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
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
                                          std::mt19937* rng, bool require_non_empty) {
  std::vector<bst_feature_t> features;
  features.reserve(n_features);
  for (bst_feature_t f = 0; f < n_features; ++f) {
    if (std::generate_canonical<float, 24>(*rng) <= fraction) {
      features.push_back(f);
    }
  }
  if (require_non_empty && features.empty() && n_features > 0) {
    std::uniform_int_distribution<bst_feature_t> dist(0, n_features - 1);
    features.push_back(dist(*rng));
  }
  return features;
}

std::vector<bst_feature_t> PredFeatures(bst_feature_t n_features, TrainParam const& param,
                                        std::vector<bst_feature_t> const& cond_features,
                                        std::mt19937* rng) {
  std::vector<bst_feature_t> pred;
  if (param.metagb_leaf_only_use_bias) {
    pred.push_back(n_features);
    return pred;
  }
  if (param.metagb_cond_pred_share) {
    pred = cond_features;
  } else {
    pred = SampleFeatures(n_features, param.metagb_pred_colsample_bytree, rng, false);
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
  std::unordered_map<bst_node_t, std::vector<bst_node_t>> node_rows_;
  std::vector<float> raw_values_;
  std::vector<bst_bin_t> digitized_bins_;
  std::vector<double> bins_;
  std::size_t n_rows_{0};
  bst_feature_t n_features_{0};
  int max_bin_{0};
  std::mt19937 rng_{1};

  [[nodiscard]] bool HasSplit(bst_node_t nidx) const {
    return split_info_.find(nidx) != split_info_.cend();
  }

  [[nodiscard]] std::vector<bst_node_t> NodeRows(bst_node_t nidx) const {
    auto it = node_rows_.find(nidx);
    CHECK(it != node_rows_.cend()) << "Missing MetaGB rows for node: " << nidx;
    return it->second;
  }

  [[nodiscard]] std::size_t BinOffset(bst_bin_t bin, bst_feature_t feature) const {
    return static_cast<std::size_t>(bin) * static_cast<std::size_t>(n_features_ + 1) + feature;
  }

  [[nodiscard]] float RawValue(std::size_t ridx, bst_feature_t feature) const {
    CHECK_LT(ridx, n_rows_);
    CHECK_LT(feature, n_features_);
    return raw_values_[ridx * n_features_ + feature];
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

  void InitOriginalBins(SparsePage const& page, std::size_t n_rows, bst_feature_t n_features,
                        int max_bin) {
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

  MetaGBStats LeafStats(linalg::MatrixView<GradientPair const> gpair,
                        std::vector<bst_node_t> const& rows, bst_feature_t pred_feature,
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
                std::vector<bst_node_t> const& rows,
                std::vector<bst_feature_t> const& pred_features, bst_feature_t n_features,
                bst_feature_t* pred_feature, float* weight) const {
    double best_score = -std::numeric_limits<double>::infinity();
    *pred_feature = n_features;
    *weight = 0.0f;
    for (auto pf : pred_features) {
      auto stats = LeafStats(gpair, rows, pf, n_features);
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
                               std::vector<bst_feature_t> const& pred_features,
                               std::mt19937* rng) {
    auto rows = NodeRows(nidx);
    CPUExpandEntry best{nidx, depth};
    if (rows.empty()) {
      return best;
    }

    bst_feature_t best_leaf_feature;
    float best_leaf_weight;
    BestLeaf(gpair, rows, pred_features, p_tree->NumFeatures(), &best_leaf_feature,
             &best_leaf_weight);
    (*p_tree)[nidx].SetLeaf(param_->learning_rate * best_leaf_weight);
    p_tree->SetMetaGBLeaf(nidx, best_leaf_feature);

    if (rows.size() <= static_cast<std::size_t>(std::max(0.0f, param_->min_child_weight))) {
      return best;
    }
    if (param_->max_depth > 0 && depth >= param_->max_depth) {
      return best;
    }

    double best_gain = -std::numeric_limits<double>::infinity();
    MetaGBSplitInfo best_meta;
    GradStats best_left_sum;
    GradStats best_right_sum;

    auto n_features = p_tree->NumFeatures();

    for (auto cf : cond_features) {
      bst_bin_t n_bins = static_cast<bst_bin_t>(max_bin_);
      if (n_bins <= 1) {
        continue;
      }
      std::vector<std::vector<MetaGBStats>> hists(pred_features.size());
      for (std::size_t pi = 0; pi < pred_features.size(); ++pi) {
        auto pf = pred_features[pi];
        hists[pi].resize(n_bins);
        for (auto ridx : rows) {
          auto gp = gpair(ridx, 0);
          if (gp.GetHess() <= 0.0f) {
            continue;
          }
          auto local_bin = OriginalBin(ridx, cf);
          if (local_bin < 0 || local_bin >= n_bins) {
            continue;
          }
          auto x = TrainValue(ridx, pf);
          hists[pi][local_bin].Add(gp.GetGrad(), gp.GetHess(), x);
        }
      }

      std::vector<std::vector<MetaGBBinSplit>> bin_splits(pred_features.size());
      for (std::size_t pi = 0; pi < pred_features.size(); ++pi) {
        auto const& hist = hists[pi];
        bin_splits[pi].resize(n_bins);
        MetaGBStats total;
        for (auto const& s : hist) {
          total.grad_x += s.grad_x;
          total.hess_xx += s.hess_xx;
          total.hess += s.hess;
          total.count += s.count;
        }
        auto base_score = Score(total, param_->metagb_lambda_l2);
        MetaGBStats left;
        for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
          left.grad_x += hist[b].grad_x;
          left.hess_xx += hist[b].hess_xx;
          left.hess += hist[b].hess;
          left.count += hist[b].count;
          auto right = total - left;
          if (left.count == 0 || right.count == 0) {
            continue;
          }
          auto gain = Score(left, param_->metagb_lambda_l2) +
                      Score(right, param_->metagb_lambda_l2) - base_score;
          bin_splits[pi][b] = MetaGBBinSplit{gain, left, right, true};
          if (param_->metagb_score_agg == 0 && gain > best_gain) {
            best_gain = gain;
            best.split.Update(static_cast<float>(gain), cf, static_cast<float>(SplitValue(b, cf)),
                              false, false,
                              GradStats{0.0, left.hess}, GradStats{0.0, right.hess});
            best_left_sum = GradStats{0.0, left.hess};
            best_right_sum = GradStats{0.0, right.hess};
            best_meta.pred_feature = pred_features[pi];
            best_meta.split_bin = b;
            best_meta.left_weight = Weight(left, param_->metagb_lambda_l2);
            best_meta.right_weight = Weight(right, param_->metagb_lambda_l2);
          }
        }
      }

      if (param_->metagb_score_agg == 0) {
        continue;
      }

      for (bst_bin_t b = 0; b + 1 < n_bins; ++b) {
        if (!bin_splits.front()[b].valid) {
          continue;
        }
        double gain = -std::numeric_limits<double>::infinity();
        if (param_->metagb_score_agg == 1) {
          gain = 0.0;
          for (auto const& pred_split : bin_splits) {
            gain += pred_split[b].gain / static_cast<double>(pred_features.size());
          }
        } else if (param_->metagb_score_agg == 2) {
          std::vector<double> scores;
          scores.reserve(pred_features.size());
          for (auto const& pred_split : bin_splits) {
            scores.push_back(pred_split[b].gain);
          }
          gain = Median(&scores);
        } else if (param_->metagb_score_agg == 3) {
          std::uniform_int_distribution<std::size_t> dist(0, pred_features.size() - 1);
          gain = bin_splits[dist(*rng)][b].gain;
        }
        if (gain > best_gain) {
          auto const& split_for_weight = bin_splits.front()[b];
          best_gain = gain;
          best.split.Update(static_cast<float>(gain), cf, static_cast<float>(SplitValue(b, cf)),
                            false, false,
                            GradStats{0.0, split_for_weight.left.hess},
                            GradStats{0.0, split_for_weight.right.hess});
          best_left_sum = GradStats{0.0, split_for_weight.left.hess};
          best_right_sum = GradStats{0.0, split_for_weight.right.hess};
          best_meta.pred_feature = pred_features.front();
          best_meta.split_bin = b;
          best_meta.left_weight = Weight(split_for_weight.left, param_->metagb_lambda_l2);
          best_meta.right_weight = Weight(split_for_weight.right, param_->metagb_lambda_l2);
        }
      }
    }

    if (best_gain == -std::numeric_limits<double>::infinity() ||
        best_gain < param_->metagb_min_gain_to_split) {
      return CPUExpandEntry{nidx, depth};
    }
    best.split.left_sum = best_left_sum;
    best.split.right_sum = best_right_sum;
    split_info_[nidx] = best_meta;
    return best;
  }

  void ApplyTreeSplit(CPUExpandEntry const& candidate, RegTree* p_tree) {
    auto it = split_info_.find(candidate.nid);
    CHECK(it != split_info_.cend());
    auto const& meta = it->second;
    auto parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    p_tree->ExpandNode(candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
                       candidate.split.DefaultLeft(), 0.0f,
                       meta.left_weight * param_->learning_rate,
                       meta.right_weight * param_->learning_rate, candidate.split.loss_chg,
                       parent_sum.GetHess(), candidate.split.left_sum.GetHess(),
                       candidate.split.right_sum.GetHess());
    auto left = p_tree->LeftChild(candidate.nid);
    auto right = p_tree->RightChild(candidate.nid);
    p_tree->SetMetaGBLeaf(left, meta.pred_feature);
    p_tree->SetMetaGBLeaf(right, meta.pred_feature);

    auto parent_rows = NodeRows(candidate.nid);
    auto split_feature = candidate.split.SplitIndex();
    auto split_bin = meta.split_bin;
    auto& left_rows = node_rows_[left];
    auto& right_rows = node_rows_[right];
    left_rows.clear();
    right_rows.clear();
    left_rows.reserve(parent_rows.size());
    right_rows.reserve(parent_rows.size());
    for (auto ridx : parent_rows) {
      if (OriginalBin(ridx, split_feature) <= split_bin) {
        left_rows.push_back(ridx);
      } else {
        right_rows.push_back(ridx);
      }
    }
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
    auto const tree_view = tree.HostScView();
    bst_node_t nidx = RegTree::kRoot;
    while (!tree_view.IsLeaf(nidx)) {
      auto split_index = tree_view.SplitIndex(nidx);
      auto value = SparsePageValue(page, ridx, split_index, tree.NumFeatures());
      nidx = value <= tree_view.SplitCond(nidx) ? tree_view.LeftChild(nidx)
                                                : tree_view.RightChild(nidx);
    }
    return nidx;
  }

 public:
  explicit MetaGBHistMaker(Context const* ctx, ObjInfo const*)
      : TreeUpdater{ctx} {
    monitor_.Init(__func__);
  }

  void Configure(Args const& args) override { hist_param_.UpdateAllowUnknown(args); }

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
    CHECK(!trees.front()->IsMultiTarget()) << "MetaGB first implementation supports scalar trees.";
    CHECK_EQ(in_gpair->Grad()->Shape(1), 1U)
        << "MetaGB expects one gradient column per scalar tree. Multi-class classification must "
           "use XGBoost's class-wise scalar-tree path.";
    CHECK_GE(param->metagb_score_agg, 0) << "MetaGB supports score_agg in [0, 3].";
    CHECK_LE(param->metagb_score_agg, 3) << "MetaGB supports score_agg in [0, 3].";
    this->param_ = param;
    auto* p_tree = trees.front();
    p_tree->EnableMetaGB();
    split_info_.clear();
    node_rows_.clear();
    InitData(p_fmat, p_tree);

    auto h_gpair = in_gpair->Grad()->HostView();
    cpu_impl::Sampler sampler{*param};
    sampler.Sample(ctx_, h_gpair);

    auto cond_features = SampleFeatures(p_tree->NumFeatures(), param->colsample_bytree, &rng_, true);
    auto pred_features = PredFeatures(p_tree->NumFeatures(), *param, cond_features, &rng_);

    auto const& page = *(p_fmat->GetBatches<SparsePage>().begin());
    CHECK_EQ(page.Size(), h_gpair.Shape(0))
        << "MetaGB initial implementation expects a single in-memory row page.";
    InitOriginalBins(page, page.Size(), p_tree->NumFeatures(), param->max_bin);
    auto& root_rows = node_rows_[RegTree::kRoot];
    root_rows.resize(page.Size());
    std::iota(root_rows.begin(), root_rows.end(), 0);

    auto root = FindBestSplit(h_gpair, p_tree, RegTree::kRoot, 0, cond_features, pred_features,
                              &rng_);
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
                                    &rng_);
        auto r_best = FindBestSplit(h_gpair, p_tree, right, depth, cond_features, pred_features,
                                    &rng_);
        if (HasSplit(l_best.nid)) {
          best_splits.push_back(l_best);
        }
        if (HasSplit(r_best.nid)) {
          best_splits.push_back(r_best);
        }
      }
      expand_set = std::move(best_splits);
    }

    auto& h_out_position = out_position[0].HostVector();
    h_out_position.resize(h_gpair.Shape(0));
    auto const& cache_page = *(p_fmat->GetBatches<SparsePage>().begin());
    common::ParallelFor(h_gpair.Shape(0), ctx_->Threads(), [&](std::size_t ridx) {
      h_out_position[ridx] = SamplePosition::Encode(PredictLeaf(cache_page, ridx, *p_tree),
                                                    h_gpair(ridx, 0).GetHess() > 0.0f);
    });
  }

  bool UpdatePredictionCache(DMatrix const* p_fmat,
                             common::Span<HostDeviceVector<bst_node_t>> node_position,
                             linalg::MatrixView<float> out_preds) override {
    if (!p_last_fmat_ || !p_last_tree_ || p_fmat != p_last_fmat_ || node_position.size() != 1) {
      return false;
    }
    CHECK(out_preds.Device().IsCPU());
    CHECK_EQ(out_preds.Shape(1), 1U);
    auto const position = node_position.front().ConstHostSpan();
    CHECK_EQ(out_preds.Shape(0), position.size());
    auto const& tree = *p_last_tree_;
    auto const& page = *(p_last_fmat_->GetBatches<SparsePage>().begin());
    common::ParallelFor(out_preds.Shape(0), ctx_->Threads(), [&](std::size_t ridx) {
      auto nidx = PredictLeaf(page, ridx, tree);
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

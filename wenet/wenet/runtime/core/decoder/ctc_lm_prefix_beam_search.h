// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef DECODER_CTC_LM_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_LM_PREFIX_BEAM_SEARCH_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>

#include "onnxruntime_cxx_api.h"
#include "decoder/onnx_asr_model.h"

#include "decoder/context_graph.h"
#include "decoder/search_interface.h"
#include "utils/utils.h"
#include "fst/symbol-table.h"


namespace wenet {


struct CtcLmPrefixBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
};

struct LmPrefixScore {
  float s = -kFloatMax;               // blank ending score
  float ns = -kFloatMax;              // none blank ending score
  float v_s = -kFloatMax;             // viterbi blank ending score
  float v_ns = -kFloatMax;            // viterbi none blank ending score
  float cur_token_prob = -kFloatMax;  // prob of current token
  std::vector<int> times_s;           // times of viterbi blank path
  std::vector<int> times_ns;          // times of viterbi none blank path
  std::vector<int> times_;

  float score() const { return LogAdd(s, ns); }
  float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
  const std::vector<int>& times() const{
//    return v_s > v_ns ? times_s : times_ns;
    return times_;
  }

//langure model
  float lm_score_ = 0;
  string prefix_text_;
  std::vector<int> unconfident_ids_;
  float lm_weight_ = 0.9;


//context
  bool has_context = false;
  int context_state = 0;
  float context_score = 0;
  std::vector<int> start_boundaries;
  std::vector<int> end_boundaries;

  void CopyContext(const LmPrefixScore& prefix_score) {
    context_state = prefix_score.context_state;
    context_score = prefix_score.context_score;
    start_boundaries = prefix_score.start_boundaries;
    end_boundaries = prefix_score.end_boundaries;
  }

  void UpdateContext(const std::shared_ptr<ContextGraph>& context_graph,
                     const LmPrefixScore& prefix_score, int word_id,
                     int prefix_len) {
    this->CopyContext(prefix_score);

    float score = 0;
    bool is_start_boundary = false;
    bool is_end_boundary = false;

    context_state =
        context_graph->GetNextState(prefix_score.context_state, word_id, &score,
                                    &is_start_boundary, &is_end_boundary);
    context_score += score;
    if (is_start_boundary) start_boundaries.emplace_back(prefix_len);
    if (is_end_boundary) end_boundaries.emplace_back(prefix_len);
  }

  float total_score() const { return score() / times().size() + context_score; }

  float total_lm_score(int lm_weight=0.9) const { return lm_weight_*lm_score_ + (1-lm_weight_)*total_score(); }
};

struct LmPrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // here we use KB&DR hash code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
    }
    return hash_code;
  }
};

class CtcLmPrefixBeamSearch : public SearchInterface {
 public:
  explicit CtcLmPrefixBeamSearch(
      const CtcLmPrefixBeamSearchOptions& opts,
      const std::shared_ptr<ContextGraph>& context_graph = nullptr,
      const std::shared_ptr<fst::SymbolTable>& unit_table = nullptr,
      const std::shared_ptr<OnnxLmModel>& lm_model = nullptr);

  void Search(const std::vector<std::vector<float>>& logp) override;
  void Reset() override;
  void FinalizeSearch() override;
  SearchType Type() const override { return SearchType::kPrefixBeamSearch; }
  void UpdateOutputs(const std::pair<std::vector<int>, LmPrefixScore>& prefix);
  void UpdateHypotheses(
      const std::vector<std::pair<std::vector<int>, LmPrefixScore>>& hpys);
  void UpdateFinalContext();

  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }
  const std::vector<std::vector<int>>& Inputs() const override {
    return hypotheses_;
  }
  const std::vector<std::vector<int>>& Outputs() const override {
    return outputs_;
  }
  const std::vector<float>& Likelihood() const override { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const override { return times_; }

  const std::vector<float>& LmScore() const override { return lm_score_; }
  void UpdateLmHypeOnce(std::unordered_map<std::vector<int>, LmPrefixScore, LmPrefixHash>& next_hyps,
                        bool lm_trigger);
  bool LmCheck(const std::vector<float>& topk_score, const std::vector<int32_t>& topk_index,
                int& cur_beam_size, bool& cur_unconfident_flag);
  void ResetLm();

 private:
  int abs_time_step_ = 0;

  // N-best list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;

  std::unordered_map<std::vector<int>, LmPrefixScore, LmPrefixHash> cur_hyps_;
  std::shared_ptr<ContextGraph> context_graph_ = nullptr;
  std::shared_ptr<fst::SymbolTable> unit_table_ = nullptr;

  std::shared_ptr<OnnxLmModel> lm_model_ = nullptr;
  std::vector<float> lm_score_;
  std::vector<string> lm_text_;

  int blank_remain_times_ = 0;
  int confident_remain_times_ = 0;
  int unconfident_cnt_ = 0;
  float confidence_threshold_ = 0.9;
  float min_candidate_prob_ = 0.01;
  float lm_weight_ = 0.9;
  std::vector<int> no_blank_ids_;
  bool lm_updated_ = false;
  int print_cnt_ = 0;


  // Outputs contain the hypotheses_ and tags like: <context> and </context>
  std::vector<std::vector<int>> outputs_;
  const CtcLmPrefixBeamSearchOptions& opts_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(CtcLmPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_

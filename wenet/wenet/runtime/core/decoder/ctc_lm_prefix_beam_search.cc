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


#include "decoder/ctc_lm_prefix_beam_search.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

CtcLmPrefixBeamSearch::CtcLmPrefixBeamSearch(
    const CtcLmPrefixBeamSearchOptions& opts,
    const std::shared_ptr<ContextGraph>& context_graph,
    const std::shared_ptr<fst::SymbolTable>& unit_table,
    const std::shared_ptr<OnnxLmModel>& lm_model)
    : opts_(opts), context_graph_(context_graph),unit_table_(unit_table),lm_model_(lm_model) {

  Reset();
}

void CtcLmPrefixBeamSearch::ResetLm(){
  lm_score_.clear();
  lm_text_.clear();

  no_blank_ids_.clear();
  blank_remain_times_ = 0;
  confident_remain_times_ = 0;
  unconfident_cnt_ = 0;
  lm_updated_ = false;
}

void CtcLmPrefixBeamSearch::Reset() {
  hypotheses_.clear();
  likelihood_.clear();
  cur_hyps_.clear();
  viterbi_likelihood_.clear();
  times_.clear();
  outputs_.clear();
  abs_time_step_ = 0;
  LmPrefixScore prefix_score;
  prefix_score.s = 0.0;
  prefix_score.ns = -kFloatMax;
  prefix_score.v_s = 0.0;
  prefix_score.v_ns = 0.0;
  std::vector<int> empty;
  cur_hyps_[empty] = prefix_score;
  outputs_.emplace_back(empty);
  hypotheses_.emplace_back(empty);
  likelihood_.emplace_back(prefix_score.total_score());
  times_.emplace_back(empty);

  ResetLm();
}

static bool PrefixScoreCompare(
    const std::pair<std::vector<int>, LmPrefixScore>& a,
    const std::pair<std::vector<int>, LmPrefixScore>& b) {
  return a.second.total_score() > b.second.total_score();
}

static bool LmPrefixScoreCompare(
    const std::pair<std::vector<int>, LmPrefixScore>& a,
    const std::pair<std::vector<int>, LmPrefixScore>& b) {
  return a.second.total_lm_score() > b.second.total_lm_score();
}


void CtcLmPrefixBeamSearch::UpdateOutputs(
    const std::pair<std::vector<int>, LmPrefixScore>& prefix) {
  const std::vector<int>& input = prefix.first;
  const std::vector<int>& start_boundaries = prefix.second.start_boundaries;
  const std::vector<int>& end_boundaries = prefix.second.end_boundaries;

  std::vector<int> output;
  int s = 0;
  int e = 0;
  for (int i = 0; i < input.size(); ++i) {
    if (s < start_boundaries.size() && i == start_boundaries[s]) {
      output.emplace_back(context_graph_->start_tag_id());
      ++s;
    }
    output.emplace_back(input[i]);
    if (e < end_boundaries.size() && i == end_boundaries[e]) {
      output.emplace_back(context_graph_->end_tag_id());
      ++e;
    }
  }
  outputs_.emplace_back(output);
}

void CtcLmPrefixBeamSearch::UpdateHypotheses(
    const std::vector<std::pair<std::vector<int>, LmPrefixScore>>& hpys) {
  cur_hyps_.clear();
  outputs_.clear();
  hypotheses_.clear();
  likelihood_.clear();
  viterbi_likelihood_.clear();
  times_.clear();

  lm_score_.clear();
  lm_text_.clear();

  for (auto& item : hpys) {
    cur_hyps_[item.first] = item.second;
    UpdateOutputs(item);
    hypotheses_.emplace_back(std::move(item.first));
    likelihood_.emplace_back(item.second.total_lm_score());
    viterbi_likelihood_.emplace_back(item.second.viterbi_score());
    times_.emplace_back(item.second.times());

    string text;
    for (size_t j = 0; j < item.first.size(); j++) {
      std::string word = unit_table_->Find(item.first[j]);
      text += (' ' + word);
    }
    lm_text_.emplace_back(text);
    lm_score_.emplace_back(item.second.lm_score_);
  }

}


void CtcLmPrefixBeamSearch::UpdateLmHypeOnce(std::unordered_map<std::vector<int>, LmPrefixScore, LmPrefixHash>& next_hyps,
                                        bool lm_trigger) {
    if (lm_trigger){
            VLOG(4)  << "lm_trigger" << std::endl;
            unconfident_cnt_ = 0;
            lm_updated_ = true;
            std::vector<float> cur_lm_score;
            std::vector<std::pair<std::string, std::vector<int>>> cur_lm_text;
            std::vector<std::vector<int>> cur_prefix;

            for (auto& it : next_hyps) {
                const std::vector<int>& prefix = it.first;
                LmPrefixScore& prefix_score = it.second;

                string cur_text;
                for (size_t j = 0; j < prefix.size(); j++) {
                  std::string word = unit_table_->Find(prefix[j]);
                  cur_text += (' ' + word);
                }
                prefix_score.prefix_text_ = cur_text;
                cur_lm_text.emplace_back(std::make_pair(prefix_score.prefix_text_, prefix_score.unconfident_ids_));
                cur_prefix.emplace_back(prefix);
            }
            lm_model_->UpdateLmScore(cur_lm_text, cur_lm_score);
            for (int i = 0; i < cur_lm_score.size(); i++) {
                const std::vector<int>& prefix = cur_prefix[i];
                LmPrefixScore& prefix_score = next_hyps[prefix];
                prefix_score.lm_score_ = cur_lm_score[i];
                prefix_score.unconfident_ids_.clear();
            }

        }

    // 3. Second beam prune, only keep top n best paths
    std::vector<std::pair<std::vector<int>, LmPrefixScore>> arr(next_hyps.begin(), next_hyps.end());

    int second_beam_size = lm_trigger ? std::min(1, opts_.second_beam_size) : std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
    std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(), LmPrefixScoreCompare);
    arr.resize(second_beam_size);
    std::sort(arr.begin(), arr.end(), LmPrefixScoreCompare);


    // 4. Update cur_hyps_ and get new result
    UpdateHypotheses(arr);

}

bool CtcLmPrefixBeamSearch::LmCheck(const std::vector<float>& topk_score, const std::vector<int32_t>& topk_index,
                int& cur_beam_size, bool& cur_unconfident_flag){

    int cur_top_id = topk_index[0];
    float cur_top_score = topk_score[0];
    if (cur_top_id == opts_.blank){
        blank_remain_times_ += 1;
    }
    else{
        no_blank_ids_.emplace_back(cur_top_id);
        blank_remain_times_ = 0;
        lm_updated_ = false;
    }

    if (cur_top_id != opts_.blank && exp(cur_top_score) < confidence_threshold_){
        cur_beam_size = 10;
        for (int i=1; i < cur_beam_size && i < topk_score.size(); i++){
            if (exp(topk_score[i]) < min_candidate_prob_ || topk_index[i] == opts_.blank ){
                cur_beam_size = i;
                break;
            }
        }

        cur_unconfident_flag = true;
        confident_remain_times_ = 0;
        unconfident_cnt_ += 1;
    }
    else{
        cur_beam_size = 1;
        if (cur_top_id != opts_.blank){
            confident_remain_times_ += 1;
        }
    }

//    print_cnt_ += 1;
//    if(print_cnt_ >= 1) {
//        print_cnt_ = 0;
//        VLOG(4) << "  " << blank_remain_times_ << "  " << no_blank_ids_.size() << "  " << confident_remain_times_ << "  " << unconfident_cnt_ << "  " << lm_updated_ << std::endl;
//    }

    bool lm_trigger = blank_remain_times_ >= 4 && no_blank_ids_.size() > 4 &&
                      confident_remain_times_ >= 3 && unconfident_cnt_ >= 2 && !lm_updated_;
    return lm_trigger;

}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in
// it.
void CtcLmPrefixBeamSearch::Search(const std::vector<std::vector<float>>& logp) {
  if (logp.size() == 0) {
    UpdateLmHypeOnce(cur_hyps_, true);
    return;
  }

  int first_beam_size =
      std::min(static_cast<int>(logp[0].size()), opts_.first_beam_size);

  for (int t = 0; t < logp.size(); ++t, ++abs_time_step_) {
    const std::vector<float>& logp_t = logp[t];
    std::unordered_map<std::vector<int>, LmPrefixScore, LmPrefixHash> next_hyps;

    // 1. First beam prune, only select topk candidates
    std::vector<float> topk_score;
    std::vector<int32_t> topk_index;
    TopK(logp_t, first_beam_size, &topk_score, &topk_index);

    if (static_cast<int>(topk_score.size()) < 1){
        continue;
    }

    int cur_beam_size = first_beam_size;
    bool cur_unconfident_flag = false;
    bool lm_trigger = LmCheck(topk_score, topk_index, cur_beam_size, cur_unconfident_flag);
//    cur_beam_size = first_beam_size;

    if (cur_unconfident_flag){
        cur_beam_size = std::min(static_cast<int>(topk_index.size()), cur_beam_size);
        for (int i = 0; i < cur_beam_size ; ++i){
            std::cout << unit_table_->Find(topk_index[i]) << " " << exp(topk_score[i]) << "    ";
        }
        std::cout << abs_time_step_ << std::endl;
    }


    for (int i = 0; i < cur_beam_size; ++i) {
          int id = topk_index[i];
          auto prob = topk_score[i];
          std::string word = unit_table_->Find(id);

          if (id == opts_.blank && cur_unconfident_flag){
            continue;
          }

          for (auto& it : cur_hyps_) {
            const std::vector<int>& prefix = it.first;
            LmPrefixScore& prefix_score = it.second;
            // If prefix doesn't exist in next_hyps, next_hyps[prefix] will insert
            // PrefixScore(-inf, -inf) by default, since the default constructor
            // of PrefixScore will set fields s(blank ending score) and
            // ns(none blank ending score) to -inf, respectively.
            if (id == opts_.blank) {
              // Case 0: *a + ε => *a
              LmPrefixScore& next_score = next_hyps[prefix];
              next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);

//              next_score.v_s = prefix_score.viterbi_score() + prob;
//              next_score.times_s = prefix_score.times();
              next_score.times_ = prefix_score.times_;

              next_score.lm_score_ = prefix_score.lm_score_;
              next_score.unconfident_ids_ = prefix_score.unconfident_ids_;

              // Prefix not changed, copy the context from prefix.
              if (context_graph_ && !next_score.has_context) {
                next_score.CopyContext(prefix_score);
                next_score.has_context = true;
              }
            }
            else if (!prefix.empty() && id == prefix.back()) {
              // Case 1: *a + a => *a
              LmPrefixScore& next_score1 = next_hyps[prefix];
              next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);

              next_score1.lm_score_ = prefix_score.lm_score_;
              next_score1.unconfident_ids_ = prefix_score.unconfident_ids_;

              next_score1.times_ = prefix_score.times_;
              next_score1.times_.back() = abs_time_step_;
//              if (next_score1.v_ns < prefix_score.v_ns + prob) {
//                next_score1.v_ns = prefix_score.v_ns + prob;
//                if (next_score1.cur_token_prob < prob) {
//                  next_score1.cur_token_prob = prob;
//                  next_score1.times_ns = prefix_score.times_ns;
//                  CHECK_GT(next_score1.times_ns.size(), 0);
//                  next_score1.times_ns.back() = abs_time_step_;
//                }
//              }
              if (context_graph_ && !next_score1.has_context) {
                next_score1.CopyContext(prefix_score);
                next_score1.has_context = true;
              }

              // Case 2: *aε + a => *aa
              std::vector<int> new_prefix(prefix);
              new_prefix.emplace_back(id);
              LmPrefixScore& next_score2 = next_hyps[new_prefix];
              next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob);

              next_score2.lm_score_ = prefix_score.lm_score_;
              next_score2.unconfident_ids_ = prefix_score.unconfident_ids_;
              if (cur_unconfident_flag){
                next_score2.unconfident_ids_.emplace_back(static_cast<int>(prefix.size()));
              }


              next_score2.times_ = prefix_score.times_;
              next_score2.times_.emplace_back(abs_time_step_);
//              if (next_score2.v_ns < prefix_score.v_s + prob) {
//                next_score2.v_ns = prefix_score.v_s + prob;
//                next_score2.cur_token_prob = prob;
//                next_score2.times_ns = prefix_score.times_s;
//                next_score2.times_ns.emplace_back(abs_time_step_);
//              }
              if (context_graph_ && !next_score2.has_context) {
                // Prefix changed, calculate the context score.
                next_score2.UpdateContext(context_graph_, prefix_score, id,
                                          prefix.size());
                next_score2.has_context = true;
              }
            }
            else {
              // Case 3: *a + b => *ab, *aε + b => *ab
              std::vector<int> new_prefix(prefix);
              new_prefix.emplace_back(id);
              LmPrefixScore& next_score = next_hyps[new_prefix];
              next_score.ns = LogAdd(next_score.ns, prefix_score.score() + prob);

              next_score.lm_score_ = prefix_score.lm_score_;
              next_score.unconfident_ids_ = prefix_score.unconfident_ids_;
              if (cur_unconfident_flag){
                next_score.unconfident_ids_.emplace_back(static_cast<int>(prefix.size()));
              }


              next_score.times_ = prefix_score.times_;
              next_score.times_.emplace_back(abs_time_step_);
//              if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
//                next_score.v_ns = prefix_score.viterbi_score() + prob;
//                next_score.cur_token_prob = prob;
//                next_score.times_ns = prefix_score.times();
//                next_score.times_ns.emplace_back(abs_time_step_);
//              }
              if (context_graph_ && !next_score.has_context) {
                // Calculate the context score.
                next_score.UpdateContext(context_graph_, prefix_score, id,
                                         prefix.size());
                next_score.has_context = true;
              }
            }
          }
        }


    lm_trigger = lm_trigger;
    UpdateLmHypeOnce(next_hyps, lm_trigger);

  }
}

void CtcLmPrefixBeamSearch::FinalizeSearch() { UpdateFinalContext(); }

void CtcLmPrefixBeamSearch::UpdateFinalContext() {
  if (context_graph_ == nullptr) return;
  CHECK_EQ(hypotheses_.size(), cur_hyps_.size());
  CHECK_EQ(hypotheses_.size(), likelihood_.size());
  // We should backoff the context score/state when the context is
  // not fully matched at the last time.
  for (const auto& prefix : hypotheses_) {
    LmPrefixScore& prefix_score = cur_hyps_[prefix];
    if (prefix_score.context_state != 0) {
      prefix_score.UpdateContext(context_graph_, prefix_score, 0,
                                 prefix.size());
    }
  }
  std::vector<std::pair<std::vector<int>, LmPrefixScore>> arr(cur_hyps_.begin(),
                                                            cur_hyps_.end());
  std::sort(arr.begin(), arr.end(), LmPrefixScoreCompare);

  // Update cur_hyps_ and get new result
  UpdateHypotheses(arr);
}

}  // namespace wenet
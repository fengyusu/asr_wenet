// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
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


#ifndef DECODER_ONNX_ASR_MODEL_H_
#define DECODER_ONNX_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>
#include <numeric>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "decoder/asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"
#include "fst/symbol-table.h"

#include "lm/model.hh"
#include "lm/config.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "util/string_stream.hh"
using namespace lm::ngram;


namespace wenet {

struct LmScore{
  int raw_idx = 0;
  float ngram_score = -kFloatMax;
  float bert_score = -kFloatMax;
  std::vector<float> bert_score_vec;
  std::vector<std::string> raw_text;
  std::vector<int> unconfident_ids;

  float get_ngram_score() const { return ngram_score; }
  float get_bert_score() const { return bert_score; }
  float get_score() const { return bert_score; }
};

class OnnxLmModel {
 public:
  // Note: Do not call the InitEngineThreads function more than once.
  static void InitEngineThreads(int num_threads = 1);
 public:
  OnnxLmModel() = default;
//  OnnxLmModel(const OnnxLmModel& other);
  void Read(const std::string& model_dir, bool use_quant_model);

  void UpdateLmScore(const std::vector<std::pair<std::string, std::vector<int>>>& texts, std::vector<float>& lm_score);

  void ForwardDetectFunc(std::vector<LmScore>& lm_scores, int filter_num);

  void GetInputOutputInfo(const std::shared_ptr<Ort::Session>& session,
                          std::vector<const char*>* in_names,
                          std::vector<const char*>* out_names);

 private:
  int unk_id_ = 100;
  int cls_id_ = 101;
  int sep_id_ = 102;
  int pad_id_ = 0;

  std::shared_ptr<Model> klm_model_ = nullptr;


  std::map<string, int64_t> lm_vocab_table_;
  // sessions
  // NOTE(Mddct): The Env holds the logging state used by all other objects.
  //  One Env must be created before using any other Onnxruntime functionality.
//  static Ort::Env lm_env_;  // shared environment across threads.
//  static Ort::SessionOptions lm_session_options_;
  std::shared_ptr<Ort::Session> lm_session_ = nullptr;

  // node names
  std::vector<const char*> lm_in_names_, lm_out_names_;


};


class OnnxAsrModel : public AsrModel {
 public:
  // Note: Do not call the InitEngineThreads function more than once.
  static void InitEngineThreads(int num_threads = 1);

 public:
  OnnxAsrModel() = default;
  OnnxAsrModel(const OnnxAsrModel& other);
  void Read(const std::string& model_dir, bool use_quant_model);
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;
  void GetInputOutputInfo(const std::shared_ptr<Ort::Session>& session,
                          std::vector<const char*>* in_names,
                          std::vector<const char*>* out_names);

 protected:
  void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
                          std::vector<std::vector<float>>* ctc_prob) override;

  float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);

 private:
  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int head_ = 0;

  // sessions
  // NOTE(Mddct): The Env holds the logging state used by all other objects.
  //  One Env must be created before using any other Onnxruntime functionality.
//  static Ort::Env env_;  // shared environment across threads.
//  static Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
  std::shared_ptr<Ort::Session> rescore_session_ = nullptr;
  std::shared_ptr<Ort::Session> ctc_session_ = nullptr;

  // node names
  std::vector<const char*> encoder_in_names_, encoder_out_names_;
  std::vector<const char*> ctc_in_names_, ctc_out_names_;
  std::vector<const char*> rescore_in_names_, rescore_out_names_;

  // caches
  Ort::Value att_cache_ort_{nullptr};
  Ort::Value cnn_cache_ort_{nullptr};
  std::vector<Ort::Value> encoder_outs_;
  // NOTE: Instead of making a copy of the xx_cache, ONNX only maintains
  //  its data pointer when initializing xx_cache_ort (see https://github.com/
  //  microsoft/onnxruntime/blob/master/onnxruntime/core/framework
  //  /tensor.cc#L102-L129), so we need the following variables to keep
  //  our data "alive" during the lifetime of decoder.
  std::vector<float> att_cache_;
  std::vector<float> cnn_cache_;
};



}  // namespace wenet

#endif  // DECODER_ONNX_ASR_MODEL_H_

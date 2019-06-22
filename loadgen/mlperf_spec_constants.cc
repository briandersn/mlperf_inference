/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlperf_spec_constants.h"

namespace mlperf {
namespace spec {
namespace v0_5 {

// TODO: Finalize server latency targets.
constexpr uint64_t kServerTargetLatencyMs_Resnet50_v1_5 = 100;
constexpr uint64_t kServerTargetLatencyMs_MobileNets_v1_224 = 100;
constexpr uint64_t kServerTargetLatencyMs_SSD_ResNet34 = 100;
constexpr uint64_t kServerTargetLatencyMs_SSD_MobileNets_v1 = 100;
constexpr uint64_t kServerTargetLatencyMs_GNMT = 100;

// TODO: Finalize fixed QPS and rule out MultiStreamFree as alternative.
constexpr double kMultiStreamFixedQPS = 20.0;

constexpr double kMinPerformanceRunDurationSeconds = 60.0;
constexpr size_t kMinQueryCountSingleStream = 1024;
constexpr size_t kMinQueryCountNotSingleStream = 24576;

constexpr uint64_t kDefaultQslSeed = 0xABCD1234;
constexpr uint64_t kDefaultSampleSeed = 0x1234ABCD;
constexpr uint64_t kDefaultScheduleSeed = 0xA1B2C3D4;

TestSettings CreateCommonSettings() {
  TestSettings s;
  s.mode = TestMode::SubmissionRun;
  s.min_duration_ms = kMinPerformanceRunDurationSeconds * 1000;
  s.qsl_rng_seed = kDefaultQslSeed;
  s.sample_index_rng_seed = kDefaultSampleSeed;
  s.schedule_rng_seed = kDefaultScheduleSeed;
  return s;
}

TestSettings CreateSingleStreamSettings(Model model,
                                        uint64_t expected_latency_ns) {
  TestSettings s = CreateCommonSettings();
  s.scenario = TestScenario::SingleStream;
  s.min_query_count = kMinQueryCountSingleStream;
  s.single_stream_expected_latency_ns = expected_latency_ns;
  return s;
}

TestSettings CreateMultiStreamSettings(Model model, int samples_per_query) {
  TestSettings s = CreateCommonSettings();
  s.scenario = TestScenario::MultiStream;
  s.min_query_count = kMinQueryCountNotSingleStream;
  s.multi_stream_target_qps = kMultiStreamFixedQPS;
  s.multi_stream_samples_per_query = samples_per_query;
  s.multi_stream_max_async_queries = 1;
  s.multi_stream_target_latency_ns = 1e9 / kMultiStreamFixedQPS;
  return s;
}

TestSettings CreateServerSettings(Model model, double target_qps,
                                  bool coalesce_queries) {
  TestSettings s = CreateCommonSettings();
  s.scenario = TestScenario::Server;
  s.min_query_count = kMinQueryCountNotSingleStream;
  s.server_target_qps = target_qps;
  s.server_coalesce_queries = coalesce_queries;
  uint64_t target_latency_ms = 0;
  switch (model) {
    case Model::Resnet50_v1_5:
      target_latency_ms = kServerTargetLatencyMs_Resnet50_v1_5;
      break;
    case Model::MobileNets_v1_224:
      target_latency_ms = kServerTargetLatencyMs_MobileNets_v1_224;
      break;
    case Model::SSD_ResNet34:
      target_latency_ms = kServerTargetLatencyMs_SSD_ResNet34;
      break;
    case Model::SSD_MobileNets_v1:
      target_latency_ms = kServerTargetLatencyMs_SSD_MobileNets_v1;
      break;
    case Model::GNMT:
      target_latency_ms = kServerTargetLatencyMs_GNMT;
      break;
  }
  s.server_target_latency_ns = target_latency_ms * 1000000;

  return s;
}

TestSettings CreateOfflineSettings(Model model, double expected_qps) {
  TestSettings s = CreateCommonSettings();
  s.scenario = TestScenario::Offline;
  s.min_query_count = kMinQueryCountNotSingleStream;
  s.offline_expected_qps = expected_qps;
  return s;
}

}  // namespace v0_5
}  // namespace spec
}  // namespace mlperf

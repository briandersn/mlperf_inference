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

#ifndef MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H
#define MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H

// WARNING: Keep in mind that the exact settings to use for submission
//          purposes have not been finalized.
//
// The functions defined in this header are the only functions in the loadgen
// that are aware of the MLPerf model categories and associated constants
// for minimum runtime, minimum queries, target latencies, etc.
//
// Using the functions here is the easiest way to make sure your
// TestSettings will be valid for submission and have the most up-to-date
// requirements.

#include <stdint.h>

#include "test_settings.h"

namespace mlperf {
namespace spec {
namespace v0_5 {

enum class Model {
  Resnet50_v1_5,
  MobileNets_v1_224,
  SSD_ResNet34,
  SSD_MobileNets_v1,
  GNMT,
};

// Each of the functions below create TestSettings that will be valid
// for results submission, as long as they aren't modified before they are
// passed to mlperf::StartTest().
TestSettings CreateSingleStreamSettings(Model model,
                                        uint64_t expected_latency_ns);
TestSettings CreateMultiStreamSettings(Model model, int samples_per_query);
TestSettings CreateServerSettings(Model model, double target_qps,
                                  bool coalesce_queries);
TestSettings CreateOfflineSettings(Model model, double expected_qps);

}  // namespace v0_5
}  // namespace spec
}  // namespace mlperf

#endif  // MLPERF_LOADGEN_MLPERF_SPEC_CONSTANTS_H

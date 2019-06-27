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

#include <future>

#include "../loadgen.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"
#include "../test_settings.h"

class SystemUnderTestNull : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestNull() = default;
  ~SystemUnderTestNull() override = default;
  const std::string& Name() const override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    for (auto s : samples) {
      responses.push_back({s.id, 0, 0});
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

 private:
  std::string name_{"NullSUT"};
};

class QuerySampleLibraryNull : public mlperf::QuerySampleLibrary {
 public:
  QuerySampleLibraryNull() = default;
  ~QuerySampleLibraryNull() = default;
  const std::string& Name() const override { return name_; }

  const size_t TotalSampleCount() override { return 1024 * 1024; }

  const size_t PerformanceSampleCount() override { return 1024; }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    return;
  }

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    return;
  }

 private:
  std::string name_{"NullQSL"};
};

void TestSingleStream() {
  SystemUnderTestNull null_sut;
  QuerySampleLibraryNull null_qsl;

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = true;

  mlperf::TestSettings ts;

  mlperf::StartTest(&null_sut, &null_qsl, ts, log_settings);
}

class SystemUnderTestNullStdAsync : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestNullStdAsync() { futures_.reserve(1000000); }
  ~SystemUnderTestNullStdAsync() override = default;
  const std::string& Name() const override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    futures_.emplace_back(std::async(std::launch::async, [samples] {
      std::vector<mlperf::QuerySampleResponse> responses;
      responses.reserve(samples.size());
      for (auto s : samples) {
        responses.push_back({s.id, 0, 0});
      }
      mlperf::QuerySamplesComplete(responses.data(), responses.size());
    }));
  }
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

 private:
  std::string name_{"NullStdAsync"};
  std::vector<std::future<void>> futures_;
};

void TestServerStdAsync() {
  SystemUnderTestNullStdAsync null_std_async_sut;
  QuerySampleLibraryNull null_qsl;

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = true;
  log_settings.log_output.copy_summary_to_stdout = true;

  mlperf::TestSettings ts;
  ts.scenario = mlperf::TestScenario::Server;
  ts.server_target_qps = 2000000;
  ts.min_duration_ms = 100;

  mlperf::StartTest(&null_std_async_sut, &null_qsl, ts, log_settings);
}

class SystemUnderTestNullPool : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestNullPool() {
    samples_.reserve(kReserveSampleSize);
    next_poll_time_ = std::chrono::high_resolution_clock::now() + poll_period_;
    for (size_t i = 0; i < thread_count_; i++) {
      threads_.emplace_back(&SystemUnderTestNullPool::WorkerThread, this);
    }
  }

  ~SystemUnderTestNullPool() override {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      keep_workers_alive_ = false;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  const std::string& Name() const override { return name_; }

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::unique_lock<std::mutex> lock(mutex_);
    samples_.insert(samples_.end(), samples.begin(), samples.end());
  }

  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

 private:
  void WorkerThread() {
    std::vector<mlperf::QuerySample> my_samples;
    my_samples.reserve(kReserveSampleSize);
    std::unique_lock<std::mutex> lock(mutex_);
    while (keep_workers_alive_) {
      next_poll_time_ += poll_period_;
      auto my_wakeup_time = next_poll_time_;
      cv_.wait_until(lock, my_wakeup_time,
                     [&]() { return !keep_workers_alive_; });
      my_samples.swap(samples_);
      lock.unlock();

      mlperf::QuerySampleResponse response;
      for (auto s : my_samples) {
        response.id = s.id;
        mlperf::QuerySamplesComplete(&response, 1);
      }

      lock.lock();
      my_samples.clear();
    }
  }

  static constexpr size_t kReserveSampleSize = 1024 * 1024;
  const std::string name_{"NullPool"};
  const size_t thread_count_ = 16;
  const std::chrono::microseconds poll_period_{100};
  std::chrono::high_resolution_clock::time_point next_poll_time_;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool keep_workers_alive_ = true;
  std::vector<std::thread> threads_;

  std::vector<mlperf::QuerySample> samples_;
};

void TestServerPool() {
  SystemUnderTestNullPool null_pool;
  QuerySampleLibraryNull null_qsl;

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = true;
  log_settings.log_output.copy_summary_to_stdout = true;

  mlperf::TestSettings ts;
  ts.scenario = mlperf::TestScenario::Server;
  ts.server_target_qps = 2000000;
  ts.min_duration_ms = 100;

  mlperf::StartTest(&null_pool, &null_qsl, ts, log_settings);
}

int main(int argc, char* argv[]) {
  TestSingleStream();
  TestServerStdAsync();
  TestServerPool();
  return 0;
}

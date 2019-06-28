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
#include <list>

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

void TestSingleStream(const mlperf::LogSettings& log_settings) {
  SystemUnderTestNull null_sut;
  QuerySampleLibraryNull null_qsl;
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

void TestServerStdAsync(const mlperf::LogSettings& log_settings) {
  SystemUnderTestNullStdAsync null_std_async_sut;
  QuerySampleLibraryNull null_qsl;
  mlperf::TestSettings ts;
  ts.scenario = mlperf::TestScenario::Server;
  ts.server_target_qps = 2000000;
  ts.min_duration_ms = 100;
  mlperf::StartTest(&null_std_async_sut, &null_qsl, ts, log_settings);
}

class SystemUnderTestNullPool : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestNullPool() {
    next_poll_time_ = std::chrono::high_resolution_clock::now();
    issued_samples_queue_.emplace_back();
    issued_samples_queue_.back().reserve(kReserveSampleSize);
    for (size_t i = 0; i < kRecycleStackMinSize * 2; i++) {
      recycled_samples_stack_.emplace_back();
      recycled_samples_stack_.back().reserve(kReserveSampleSize);
    }
    for (size_t i = 0; i < kWorkerCount; i++) {
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
    bool need_new_set = false;
    std::vector<mlperf::QuerySample> set;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (issued_samples_queue_.back().size() < kReserveSampleSize) {
        set = std::move(issued_samples_queue_.back());
        issued_samples_queue_.pop_back();
      } else {
        need_new_set = true;
      }
    }

    for (auto& s : samples) {
      if (need_new_set) {
        need_new_set = false;
        std::unique_lock<std::mutex> lock(mutex_);
        set = GetNewIssueSetLocked();
      }
      set.push_back(s);
      if (set.size() >= kReserveSampleSize) {
        need_new_set = true;
        std::unique_lock<std::mutex> lock(mutex_);
        issued_samples_queue_.push_back(std::move(set));
      }
    }

    if (!need_new_set) {
      std::unique_lock<std::mutex> lock(mutex_);
      issued_samples_queue_.push_back(std::move(set));
    }

    cv_.notify_all();
  }

  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

 private:
  std::vector<mlperf::QuerySample> GetNewIssueSetLocked() {
    std::vector<mlperf::QuerySample> set;
    if (recycled_samples_stack_.empty()) {
      set.reserve(kReserveSampleSize);
    } else {
      set = std::move(recycled_samples_stack_.back());
      recycled_samples_stack_.pop_back();
    }
    return set;
  }

  void WorkerThread() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (keep_workers_alive_) {
      if (issued_samples_queue_.size() <= 1) {
        next_poll_time_ += poll_period_;
        auto my_wakeup_time = next_poll_time_;
        cv_.wait_until(lock, my_wakeup_time, [&]() {
          return !keep_workers_alive_ ||
                 (!issued_samples_queue_.empty() &&
                  !issued_samples_queue_.front().empty());
        });
      } else {
        next_poll_time_ =
            std::chrono::high_resolution_clock::now() + poll_period_;
      }

      std::vector<mlperf::QuerySample> my_samples(
          std::move(issued_samples_queue_.front()));
      issued_samples_queue_.pop_front();
      issued_samples_queue_.push_back(GetNewIssueSetLocked());
      bool recycled_stack_getting_low =
          recycled_samples_stack_.size() < kRecycleStackMinSize;
      lock.unlock();

      mlperf::QuerySampleResponse response;
      for (auto s : my_samples) {
        response.id = s.id;
        mlperf::QuerySamplesComplete(&response, 1);
      }

      std::vector<mlperf::QuerySample> more_recycled_samples;
      if (recycled_stack_getting_low) {
        more_recycled_samples.reserve(kRecycleStackMinSize);
      }

      lock.lock();
      my_samples.clear();
      recycled_samples_stack_.push_back(std::move(my_samples));
      if (recycled_stack_getting_low) {
        recycled_samples_stack_.push_back(std::move(more_recycled_samples));
      }
    }
  }

  static constexpr size_t kWorkerCount = 18;
  static constexpr size_t kReserveSampleSize = 64;
  static constexpr size_t kRecycleStackMinSize = kWorkerCount * 3;
  const std::string name_{"NullPool"};

  const std::chrono::microseconds poll_period_{100};
  std::chrono::high_resolution_clock::time_point next_poll_time_;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool keep_workers_alive_ = true;
  std::vector<std::thread> threads_;

  std::list<std::vector<mlperf::QuerySample>> issued_samples_queue_;

  std::vector<std::vector<mlperf::QuerySample>> recycled_samples_stack_;
};

void TestServerPool(const mlperf::LogSettings& log_settings) {
  SystemUnderTestNullPool null_pool;
  QuerySampleLibraryNull null_qsl;
  mlperf::TestSettings ts;
  ts.scenario = mlperf::TestScenario::Server;
  ts.server_target_qps = 6000000;
  ts.min_duration_ms = 100;
  mlperf::StartTest(&null_pool, &null_qsl, ts, log_settings);
}

int main(int argc, char* argv[]) {
  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = true;
  log_settings.log_output.copy_summary_to_stdout = true;
  log_settings.log_output.outdir = "logs";
  TestSingleStream(log_settings);
  TestServerStdAsync(log_settings);
  TestServerPool(log_settings);
  TestServerPool(log_settings);
  return 0;
}

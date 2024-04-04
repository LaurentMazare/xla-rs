#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#pragma GCC diagnostic ignored "-Wreturn-type"
#include "xla/client/client_library.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"

Shape make_shape_internal(int pr_type, int dsize, const int64_t *ds) {
  bool has_negative_dim = false;
  for (int i = 0; i < dsize; ++i) {
    if (ds[i] < 0) {
      has_negative_dim = true;
      break;
    }
  }
  Shape shape;
  if (has_negative_dim) {
    std::vector<bool> dynamic;
    std::vector<int64_t> bounds;
    for (int i = 0; i < dsize; ++i) {
      if (ds[i] < 0) {
        bounds.push_back(-ds[i]);
        dynamic.push_back(true);
      } else {
        bounds.push_back(ds[i]);
        dynamic.push_back(false);
      }
    }
    shape = ShapeUtil::MakeShape(
        (PrimitiveType)pr_type,
        absl::Span<const int64_t>(bounds.data(), bounds.size()), dynamic);
  } else {
    shape = ShapeUtil::MakeShape((PrimitiveType)pr_type,
                                 absl::Span<const int64_t>(ds, dsize));
  }
  return shape;
}

int main() {
  xla::GpuAllocatorConfig allocator = {.memory_fraction = memory_fraction,
                                       .preallocate = preallocate};
  auto client_or = xla::GetStreamExecutorGpuClient(false, allocator, 0, 0)

  if(!client_or.ok()) {
    std::cout << strdup(tsl::NullTerminatedMessage(*clientor.status)) << "\n";
    return 1;
  } else {

    auto client = client_or.ok();
    auto builder = xla::XlaBuilder("test");

    auto dims = (int64_t*)(malloc(sizeof(int64_t)));
    dims[0] = 1000000;
    auto shape = make_shape_internal(11, 1000000, dims)
    auto x = xla::Parameter(builder, 0, shape)

    auto comp_or = builder.Build(x);

    if(!comp_or.ok()) {
        std::cout << strdup(tsl::NullTerminatedMessage(*comp_or.status)) << "\n";
        return 1;
    } else {

        auto comp = comp_or.ok();
        CompileOptions options;
        auto exe_or = client.Compile(comp, options);

        if(!exe_or.ok()) {
            std::cout << strdup(tsl::NullTerminatedMessage(*exe_or.status)) << "\n";
            return 1;
        } else {

            auto exe = exe_or.ok();

            vector<f32_t> vec = {};
            for(size_t i = 0; i < 1000000; i++) {
                vec.push(0.0);
            }
            auto span = absl::Span<vector<f32_t>>(x_cpp);
            auto x_literal = LiteralUtil::CreateR1<f32_t>(span);
            auto x_buffer = client.BufferFromHostLiteral(x_literal);

            ExecuteOptions options;
            options.strict_shape_checking = false;

            while true {
                auto result_or = exe.Execute({x_buffer});
                if(!result_or.ok()) {
                    std::cout << strdup(tsl::NullTerminatedMessage(*result_or.status)) << "\n";
                    return 1;
                } else {
                    x_buffer = result_or.ok()[0][0];
                }
            }

        }
    }
  }
}
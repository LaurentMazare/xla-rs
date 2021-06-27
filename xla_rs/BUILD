# bazel build --define framework_shared_object=false tensorflow/compiler/xla/xla_rs:libxla_rs.so
package(default_visibility=["//visibility:private"])

cc_binary(
  name = "libxla_rs.so",
  srcs = ["xla_rs.cc", "xla_rs.h"],
  deps = [
    "@org_tensorflow//tensorflow/compiler/xla/client:client",
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/xla:statusor",
    "@org_tensorflow//tensorflow/compiler/xla:util",
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
  ],
  linkopts = ["-shared"],
  linkshared = 1,
)

#include "xla_rs.h"

#define ASSIGN_OR_RETURN_STATUS(lhs, rexpr) \
  ASSIGN_OR_RETURN_STATUS_IMPL(TF_STATUS_MACROS_CONCAT_NAME(_statusor, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_STATUS_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr); \
  if (!statusor.ok()) return new Status(statusor.status()); \
  auto lhs = std::move(statusor.value());

status pjrt_client_create(pjrt_client *output) {
  ASSIGN_OR_RETURN_STATUS(client, xla::GetTfrtCpuClient(false));
  *output = new std::unique_ptr(std::move(client));
  return nullptr;

}

int pjrt_client_device_count(pjrt_client c) {
  return (*c)->device_count();
}

int pjrt_client_addressable_device_count(pjrt_client c) {
  return (*c)->addressable_device_count();
}

void pjrt_client_devices(pjrt_client c, pjrt_device* outputs) {
  size_t index = 0;
  for (auto device : (*c)->devices()) {
      outputs[index++] = device;
  }
}

void pjrt_client_addressable_devices(pjrt_client c, pjrt_device* outputs) {
  size_t index = 0;
  for (auto device : (*c)->addressable_devices()) {
      outputs[index++] = device;
  }
}

char* pjrt_client_platform_name(pjrt_client c) {
  // TODO: Avoid the double allocation when converting string views.
  return strdup(std::string((*c)->platform_name()).c_str());
}

char* pjrt_client_platform_version(pjrt_client c) {
  return strdup(std::string((*c)->platform_version()).c_str());
}

void pjrt_client_free(pjrt_client b) {
  delete b;
}

void pjrt_loaded_executable_free(pjrt_loaded_executable b) {
  delete b;
}

int pjrt_device_id(pjrt_device d) {
  return d->id();
}

int pjrt_device_process_index(pjrt_device d) {
  return d->process_index();
}

int pjrt_device_local_hardware_id(pjrt_device d) {
  return d->local_hardware_id();
}

char* pjrt_device_kind(pjrt_device d) {
  return strdup(std::string(d->device_kind()).c_str());
}

char* pjrt_device_debug_string(pjrt_device d) {
  return strdup(std::string(d->DebugString()).c_str());
}

char* pjrt_device_to_string(pjrt_device d) {
  return strdup(std::string(d->ToString()).c_str());
}

xla_builder xla_builder_create(const char *name) {
  return new XlaBuilder(name);
}

void xla_builder_free(xla_builder b) {
  delete b;
}

xla_op constant_r0_float(const xla_builder b, float f) {
  return new XlaOp(ConstantR0<float>(b, f));
}

xla_op constant_r1_float(const xla_builder b, int64_t len, float f) {
  return new XlaOp(ConstantR1<float>(b, len, f));
}

xla_op parameter(const xla_builder b, int64_t id, int pr_type, int dsize, const long int *ds, const char *name) {
  return new XlaOp(Parameter(b, id, ShapeUtil::MakeShape((PrimitiveType)pr_type, absl::Span<const long int>(ds, dsize)), std::string(name)));
}

xla_op op_add(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Add(*lhs, *rhs));
}

xla_op op_sub(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Sub(*lhs, *rhs));
}

xla_op op_mul(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Mul(*lhs, *rhs));
}

xla_op op_div(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Div(*lhs, *rhs));
}

xla_op op_rem(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Rem(*lhs, *rhs));
}

xla_op op_max(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Max(*lhs, *rhs));
}

xla_op op_min(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Min(*lhs, *rhs));
}

xla_op op_and(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(And(*lhs, *rhs));
}

xla_op op_or(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Or(*lhs, *rhs));
}

xla_op op_xor(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Xor(*lhs, *rhs));
}

xla_op op_atan2(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Atan2(*lhs, *rhs));
}

xla_op op_pow(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Pow(*lhs, *rhs));
}

xla_op op_dot(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Dot(*lhs, *rhs));
}

xla_op op_not(const xla_op arg) {
  return new XlaOp(Not(*arg));
}

xla_op op_abs(const xla_op arg) {
  return new XlaOp(Abs(*arg));
}

xla_op op_exp(const xla_op arg) {
  return new XlaOp(Exp(*arg));
}

xla_op op_expm1(const xla_op arg) {
  return new XlaOp(Expm1(*arg));
}

xla_op op_floor(const xla_op arg) {
  return new XlaOp(Floor(*arg));
}

xla_op op_ceil(const xla_op arg) {
  return new XlaOp(Ceil(*arg));
}

xla_op op_round(const xla_op arg) {
  return new XlaOp(Round(*arg));
}

xla_op op_log(const xla_op arg) {
  return new XlaOp(Log(*arg));
}

xla_op op_log1p(const xla_op arg) {
  return new XlaOp(Log1p(*arg));
}

xla_op op_logistic(const xla_op arg) {
  return new XlaOp(Logistic(*arg));
}

xla_op op_sign(const xla_op arg) {
  return new XlaOp(Sign(*arg));
}

xla_op op_clz(const xla_op arg) {
  return new XlaOp(Clz(*arg));
}

xla_op op_cos(const xla_op arg) {
  return new XlaOp(Cos(*arg));
}

xla_op op_sin(const xla_op arg) {
  return new XlaOp(Sin(*arg));
}

xla_op op_tanh(const xla_op arg) {
  return new XlaOp(Tanh(*arg));
}

xla_op op_real(const xla_op arg) {
  return new XlaOp(Real(*arg));
}

xla_op op_imag(const xla_op arg) {
  return new XlaOp(Imag(*arg));
}

xla_op op_sqrt(const xla_op arg) {
  return new XlaOp(Sqrt(*arg));
}

xla_op op_rsqrt(const xla_op arg) {
  return new XlaOp(Rsqrt(*arg));
}

xla_op op_cbrt(const xla_op arg) {
  return new XlaOp(Cbrt(*arg));
}

xla_op op_is_finite(const xla_op arg) {
  return new XlaOp(IsFinite(*arg));
}

xla_op op_neg(const xla_op arg) {
  return new XlaOp(Neg(*arg));
}

int xla_op_valid(const xla_op op) {
  return op->valid();
}

void xla_op_free(xla_op o) {
  delete o;
}

int shape_dimensions_size(const shape s) {
  return s->dimensions_size();
}

int shape_element_type(const shape s) {
  return s->element_type();
}

int64_t shape_dimensions(const shape s, int i) {
  return s->dimensions(i);
}

void shape_free(shape s) {
  delete s;
}

status get_shape(const xla_builder b, const xla_op o, shape *out_shape) {
  ASSIGN_OR_RETURN_STATUS(shape, b->GetShape(*o));
  *out_shape = new Shape(shape);
  return nullptr;
}

status build(const xla_builder b, const xla_op o, xla_computation *output) {
  ASSIGN_OR_RETURN_STATUS(computation, b->Build(o));
  *output = new XlaComputation();
  **output = std::move(computation);
  return nullptr;
}

status compile(const pjrt_client client, const xla_computation computation, pjrt_loaded_executable* output) {
  CompileOptions options;
  ASSIGN_OR_RETURN_STATUS(executable, (*client)->Compile(*computation, options));
  *output = new std::unique_ptr(std::move(executable));
  return nullptr;
}

status execute(const pjrt_loaded_executable exe, const literal *inputs, int ninputs, literal *output) {
  auto client = (*exe)->client();
  ExecuteOptions options;
  std::vector<std::unique_ptr<PjRtBuffer>> input_buffers;
  std::vector<PjRtBuffer*> input_buffer_ptrs;
  PjRtDevice* device = client->devices()[0];
  for (int i = 0; i < ninputs; ++i) {
    ASSIGN_OR_RETURN_STATUS(buffer, client->BufferFromHostLiteral(*inputs[i], device));
    PjRtBuffer* buffer_ptr = buffer.get();
    input_buffers.push_back(std::move(buffer));
    input_buffer_ptrs.push_back(buffer_ptr);
  }
  ASSIGN_OR_RETURN_STATUS(
    results,
    (*exe)->Execute({input_buffer_ptrs}, options));
  ASSIGN_OR_RETURN_STATUS(literal, results[0][0]->ToLiteralSync());
  *output = new Literal();
  **output = std::move(*literal);
  return nullptr;
}

literal create_r0_f32(float f) {
    return new Literal(LiteralUtil::CreateR0<float>(f));
}

literal create_r1_f32(const float *f, int nel) {
    return new Literal(LiteralUtil::CreateR1<float>(absl::Span<const float>(f, nel)));
}

float literal_get_first_element_f32(const literal l) {
  return l->GetFirstElement<float>();
}

int64_t literal_element_count(const literal l) {
  return l->element_count();
}

void literal_shape(const literal l, shape *out_shape) {
  *out_shape = new Shape(l->shape());
}

void literal_free(literal l) {
  delete l;
}

void status_free(status s) {
  delete s;
}

void xla_computation_free(xla_computation c) {
  delete c;
}

char *status_error_message(status s) {
  return strdup(s->error_message().c_str());
}

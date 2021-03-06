#include "xla_rs.h"

#define ASSIGN_OR_RETURN_STATUS(lhs, rexpr) \
  ASSIGN_OR_RETURN_STATUS_IMPL(TF_STATUS_MACROS_CONCAT_NAME(_statusor, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_STATUS_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr); \
  if (!statusor.ok()) return new Status(statusor.status()); \
  auto lhs = std::move(statusor.ValueOrDie());

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

status transfer(const global_data gd, literal *out) {
  ASSIGN_OR_RETURN_STATUS(client, ClientLibrary::GetOrCreateLocalClient());
  ASSIGN_OR_RETURN_STATUS(literal, client->Transfer(*gd));
  *out = new Literal();
  **out = std::move(literal);
  return nullptr;
}

status transfer_to_server(const literal ls, global_data *out) {
  ASSIGN_OR_RETURN_STATUS(client, ClientLibrary::GetOrCreateLocalClient());
  ASSIGN_OR_RETURN_STATUS(global_data, client->TransferToServer(*ls));
  *out = global_data.release();
  return nullptr;
}

status build(const xla_builder b, const xla_op o, xla_computation *output) {
  ASSIGN_OR_RETURN_STATUS(computation, b->Build(o));
  *output = new XlaComputation();
  **output = std::move(computation);
  return nullptr;
}

status run(const xla_computation c, const global_data *gd, int ngd, literal *output) {
  ASSIGN_OR_RETURN_STATUS(client, ClientLibrary::GetOrCreateLocalClient());
  ASSIGN_OR_RETURN_STATUS(
    literal,
    client->ExecuteAndTransfer(*c, absl::Span<const global_data>(gd, ngd)));
  *output = new Literal();
  **output = std::move(literal);
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

void literal_free(literal l) {
  delete l;
}

void status_free(status s) {
  delete s;
}

void global_data_free(global_data gd) {
  delete gd;
}

void xla_computation_free(xla_computation c) {
  delete c;
}

char *status_error_message(status s) {
  return strdup(s->error_message().c_str());
}

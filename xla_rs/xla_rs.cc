#include "tensorflow/compiler/xla/xla_rs/xla_rs.h"

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

xla_op parameter(const xla_builder b, int64_t id, int pr_type, int dsize, const long long int *ds, const char *name) {
  return new XlaOp(Parameter(b, id, ShapeUtil::MakeShape((PrimitiveType)pr_type, absl::Span<const long long int>(ds, dsize)), std::string(name)));
}

xla_op add(const xla_op lhs, const xla_op rhs) {
  return new XlaOp(Add(*lhs, *rhs));
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

status transfer_to_server(const literal_slice ls, global_data *out) {
  ASSIGN_OR_RETURN_STATUS(client, ClientLibrary::GetOrCreateLocalClient());
  ASSIGN_OR_RETURN_STATUS(global_data, client->TransferToServer(*ls));
  *out = global_data.release();
  return nullptr;
}

status run(const xla_builder b, const xla_op o, const global_data *gd, int ngd, literal *output) {
  ASSIGN_OR_RETURN_STATUS(computation, b->Build());
  ASSIGN_OR_RETURN_STATUS(client, ClientLibrary::GetOrCreateLocalClient());
  ASSIGN_OR_RETURN_STATUS(
    literal,
    client->ExecuteAndTransfer(computation, absl::Span<const global_data>(gd, ngd)));
  *output = new Literal();
  **output = std::move(literal);
  return nullptr;
}

float literal_get_first_element_f32(const literal l) {
  return l->GetFirstElement<float>();
}

void literal_free(literal l) {
  delete l;
}

void literal_slice_free(literal_slice l) {
  delete l;
}

void status_free(status s) {
  delete s;
}

void global_data_free(global_data gd) {
  delete gd;
}

char *status_error_message(status s) {
  return strdup(s->error_message().c_str());
}

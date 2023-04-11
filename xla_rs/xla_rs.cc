#include "xla_rs.h"

#define ASSIGN_OR_RETURN_STATUS(lhs, rexpr) \
  ASSIGN_OR_RETURN_STATUS_IMPL(TF_STATUS_MACROS_CONCAT_NAME(_statusor, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_STATUS_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr); \
  if (!statusor.ok()) return new Status(statusor.status()); \
  auto lhs = std::move(statusor.value());

#define MAYBE_RETURN_STATUS(rexpr) \
  MAYBE_RETURN_STATUS_IMPL(TF_STATUS_MACROS_CONCAT_NAME(_status, __COUNTER__), rexpr)

#define MAYBE_RETURN_STATUS_IMPL(statusor, rexpr) \
  auto statusor = (rexpr); \
  if (!statusor.ok()) return new Status(statusor);

#define BEGIN_PROTECT_OP try {
#define END_PROTECT_OP_B(builder) } catch (std::exception e) { \
  return new XlaOp(builder->ReportError(tsl::errors::Internal(e.what()))); \
}
#define END_PROTECT_OP(arg) } catch (std::exception e) { \
  return new XlaOp(arg->builder()->ReportError(tsl::errors::Internal(e.what()))); \
}

status pjrt_client_create(pjrt_client *output) {
  ASSIGN_OR_RETURN_STATUS(client, xla::GetTfrtCpuClient(false));
  *output = client.release();
  return nullptr;

}

int pjrt_client_device_count(pjrt_client c) {
  return c->device_count();
}

int pjrt_client_addressable_device_count(pjrt_client c) {
  return c->addressable_device_count();
}

void pjrt_client_devices(pjrt_client c, pjrt_device* outputs) {
  size_t index = 0;
  for (auto device : c->devices()) {
      outputs[index++] = device;
  }
}

void pjrt_client_addressable_devices(pjrt_client c, pjrt_device* outputs) {
  size_t index = 0;
  for (auto device : c->addressable_devices()) {
      outputs[index++] = device;
  }
}

char* pjrt_client_platform_name(pjrt_client c) {
  // TODO: Avoid the double allocation when converting string views.
  return strdup(std::string(c->platform_name()).c_str());
}

char* pjrt_client_platform_version(pjrt_client c) {
  return strdup(std::string(c->platform_version()).c_str());
}

void pjrt_client_free(pjrt_client b) {
  delete b;
}

void pjrt_loaded_executable_free(pjrt_loaded_executable b) {
  delete b;
}

status pjrt_buffer_from_host_buffer(
    const pjrt_client client,
    const pjrt_device device,
    const void *d,
    int pr_type,
    int dsize,
    const int64_t *ds, 
    pjrt_buffer *output) {
  PjRtDevice *device_ = device == nullptr ? client->devices()[0] : device;
  ASSIGN_OR_RETURN_STATUS(buffer, client->BufferFromHostBuffer(
        d,
        (PrimitiveType)pr_type,
        absl::Span<const int64_t>(ds, dsize),
        {},
        PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
        [] () {},
        device_
  ));
  *output = buffer.release();
  return nullptr;
}

status pjrt_buffer_from_host_literal(const pjrt_client client, const pjrt_device device, const literal l, pjrt_buffer *output) {
  PjRtDevice *d = device == nullptr ? client->devices()[0] : device;
  ASSIGN_OR_RETURN_STATUS(buffer, client->BufferFromHostLiteral(*l, d));
  *output = buffer.release();
  return nullptr;
}

status pjrt_buffer_to_literal_sync(pjrt_buffer b, literal *output) {
  ASSIGN_OR_RETURN_STATUS(literal, b->ToLiteralSync());
  *output = new Literal();
  **output = std::move(*literal);
  return nullptr;
}

shape pjrt_buffer_on_device_shape(pjrt_buffer b) {
  return new Shape(b->on_device_shape());
}

status pjrt_buffer_copy_to_device(pjrt_buffer b, pjrt_device device, pjrt_buffer* output) {
  ASSIGN_OR_RETURN_STATUS(copied_b, b->CopyToDevice(device));
  *output = copied_b.release();
  return nullptr;
}

status pjrt_buffer_copy_raw_to_host_sync(pjrt_buffer b, void* dst, size_t offset, size_t transfer_size) {
  MAYBE_RETURN_STATUS(b->CopyRawToHost(dst, offset, transfer_size).Await());
  return nullptr;
}

void pjrt_buffer_free(pjrt_buffer b) {
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

xla_op constant_literal(const xla_builder b, const literal l) {
  BEGIN_PROTECT_OP
  return new XlaOp(ConstantLiteral(b, *l));
  END_PROTECT_OP_B(b)
}

#define CONST_OP_R01(native_type, primitive_type) \
  xla_op constant_r0_ ## native_type(const xla_builder b, native_type f) { \
    return new XlaOp(ConstantR0<native_type>(b, f)); \
  } \
  xla_op constant_r1c_ ## native_type(const xla_builder b, native_type f, size_t len) { \
    return new XlaOp(ConstantR1<native_type>(b, len, f)); \
  } \
  xla_op constant_r1_ ## native_type(const xla_builder b, const native_type *f, size_t len) { \
    return new XlaOp(ConstantR1<native_type>(b, absl::Span<const native_type>(f, len))); \
  } \
  literal create_r0_ ## native_type(native_type f) { \
    return new Literal(LiteralUtil::CreateR0<native_type>(f)); \
  } \
  literal create_r1_ ## native_type(const native_type *f, size_t nel) { \
    return new Literal(LiteralUtil::CreateR1<native_type>(absl::Span<const native_type>(f, nel))); \
  } \
  native_type literal_get_first_element_ ## native_type(const literal l) { \
    return l->GetFirstElement<native_type>(); \
  }

FOR_EACH_NATIVE_TYPE(CONST_OP_R01)
#undef CONST_OP_R01

xla_op parameter(const xla_builder b, int64_t id, int pr_type, int dsize, const long int *ds, const char *name) {
  BEGIN_PROTECT_OP
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
    shape = ShapeUtil::MakeShape((PrimitiveType)pr_type, absl::Span<const long int>(bounds.data(), bounds.size()), dynamic);
  } else {
    shape = ShapeUtil::MakeShape((PrimitiveType)pr_type, absl::Span<const long int>(ds, dsize));
  }
  return new XlaOp(Parameter(b, id, shape, std::string(name)));
  END_PROTECT_OP_B(b)
}

xla_op op_add(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Add(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_sub(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sub(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_mul(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Mul(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_div(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Div(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_rem(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Rem(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_max(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Max(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_min(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Min(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_and(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(And(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_or(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Or(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_xor(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Xor(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_atan2(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Atan2(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_pow(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Pow(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_dot(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Dot(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_eq(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Eq(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_ne(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Ne(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_ge(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Ge(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_gt(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Gt(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_le(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Le(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_lt(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Lt(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_not(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Not(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_abs(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Abs(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_exp(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Exp(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_expm1(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Expm1(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_floor(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Floor(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_ceil(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Ceil(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_round(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Round(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_log(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Log(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_log1p(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Log1p(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_logistic(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Logistic(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_sign(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sign(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_clz(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Clz(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_cos(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Cos(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_sin(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sin(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_tanh(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Tanh(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_real(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Real(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_imag(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Imag(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_sqrt(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sqrt(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_rsqrt(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Rsqrt(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_cbrt(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Cbrt(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_is_finite(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(IsFinite(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_neg(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Neg(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_lower_triangle(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(LowerTriangle(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_upper_triangle(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(UpperTriangle(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_einsum1(const xla_op arg, const char *config) {
  BEGIN_PROTECT_OP
  return new XlaOp(Einsum(*arg, config));
  END_PROTECT_OP(arg)
}

xla_op op_einsum2(const xla_op arg1, const xla_op arg2, const char *config) {
  BEGIN_PROTECT_OP
  return new XlaOp(Einsum(*arg1, *arg2, config));
  END_PROTECT_OP(arg1)
}

xla_op op_copy(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Copy(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_clone(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(*arg);
  END_PROTECT_OP(arg)
}

xla_op op_zeros_like(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(ZerosLike(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_zero_like(const xla_op arg) {
  BEGIN_PROTECT_OP
  const Shape* shape = arg->builder()->GetShapePtr(*arg).value();
  return new XlaOp(Zero(arg->builder(), shape->element_type()));
  END_PROTECT_OP(arg)
}

xla_op op_reshape(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Reshape(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_broadcast(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Broadcast(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_collapse(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Collapse(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_transpose(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Transpose(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_clamp(const xla_op arg1, const xla_op arg2, const xla_op arg3) {
  BEGIN_PROTECT_OP
  return new XlaOp(Clamp(*arg1, *arg2, *arg3));
  END_PROTECT_OP(arg1)
}

xla_op op_select(const xla_op arg1, const xla_op arg2, const xla_op arg3) {
  BEGIN_PROTECT_OP
  return new XlaOp(Select(*arg1, *arg2, *arg3));
  END_PROTECT_OP(arg1)
}

xla_op op_rng_uniform(const xla_op arg1, const xla_op arg2, int pr_type, int dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  auto shape = ShapeUtil::MakeShape(
      (PrimitiveType)pr_type,
      absl::Span<const long int>(ds, dsize)
  );
  return new XlaOp(RngUniform(*arg1, *arg2, shape));
  END_PROTECT_OP(arg1)
}

xla_op op_rng_normal(const xla_op arg1, const xla_op arg2, int pr_type, int dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  auto shape = ShapeUtil::MakeShape(
      (PrimitiveType)pr_type,
      absl::Span<const long int>(ds, dsize)
  );
  return new XlaOp(RngNormal(*arg1, *arg2, shape));
  END_PROTECT_OP(arg1)
}

xla_op op_slice_in_dim(const xla_op arg, int64_t start, int64_t stop, int64_t stride, int64_t dim) {
  BEGIN_PROTECT_OP
  return new XlaOp(SliceInDim(*arg, start, stop, stride, dim));
  END_PROTECT_OP(arg)
}

xla_op op_concat_in_dim(const xla_op arg, const xla_op *args, size_t nargs, int64_t dim) {
  BEGIN_PROTECT_OP
  std::vector<XlaOp> args_ = { *arg };
  for (size_t i = 0; i < nargs; ++i) {
    args_.push_back(*args[i]);
  }
  return new XlaOp(ConcatInDim(arg->builder(), absl::Span<const XlaOp>(args_), dim));
  END_PROTECT_OP(arg)
}

xla_op op_gather(
    const xla_op arg1,
    const xla_op arg2,
    const int64_t* offset_dims,
    size_t noffset_dims,
    const int64_t* collapsed_slice_dims,
    size_t ncollapsed_slice_dims,
    const int64_t* start_index_map,
    size_t nstart_index_map,
    const int64_t* set_index_vector_dim,
    const int64_t* slice_sizes,
    size_t nslice_sizes
) {
  BEGIN_PROTECT_OP
  GatherDimensionNumbers dn;
  for (size_t i = 0; i < noffset_dims; ++i) {
    dn.add_offset_dims(offset_dims[i]);
  }
  for (size_t i = 0; i < ncollapsed_slice_dims; ++i) {
    dn.add_collapsed_slice_dims(collapsed_slice_dims[i]);
  }
  for (size_t i = 0; i < nstart_index_map; ++i) {
    dn.add_start_index_map(start_index_map[i]);
  }
  if (set_index_vector_dim) {
    dn.set_index_vector_dim(*set_index_vector_dim);
  }
  auto ss = absl::Span<const int64_t>(slice_sizes, nslice_sizes);
  return new XlaOp(Gather(*arg1, *arg2, dn, ss));
  END_PROTECT_OP(arg1)
}

xla_op op_convert_element_type(const xla_op arg, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(ConvertElementType(*arg, (PrimitiveType)pr_type));
  END_PROTECT_OP(arg)
}

xla_op op_dimension_size(const xla_op arg, int64_t dim) {
  BEGIN_PROTECT_OP
  return new XlaOp(GetDimensionSize(*arg, dim));
  END_PROTECT_OP(arg)
}

xla_op op_reduce(const xla_op arg, const xla_op init, const xla_computation comp, const int64_t* dims, size_t ndims) {
  BEGIN_PROTECT_OP
  return new XlaOp(Reduce(*arg, *init, *comp, absl::Span<const int64_t>(dims, ndims)));
  END_PROTECT_OP(arg)
}

xla_op op_internal_error(const xla_builder b, const char* error) {
  BEGIN_PROTECT_OP
  return new XlaOp(b->ReportError(tsl::errors::Internal(error)));
  END_PROTECT_OP_B(b)
}

xla_op op_unknown_error(const xla_builder b, const char* error) {
  BEGIN_PROTECT_OP
  return new XlaOp(b->ReportError(tsl::errors::Unknown(error)));
  END_PROTECT_OP_B(b)
}

xla_op op_invalid_argument_error(const xla_builder b, const char* error) {
  BEGIN_PROTECT_OP
  return new XlaOp(b->ReportError(tsl::errors::InvalidArgument(error)));
  END_PROTECT_OP_B(b)
}

xla_op op_zero(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(Zero(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_op op_one(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(One(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_op op_min_value(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(MinValue(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_op op_max_value(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(MaxValue(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_builder op_builder(const xla_op arg) {
  return arg->builder();
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

status get_element_type(const xla_builder b, const xla_op o, int *out_element_type) {
  ASSIGN_OR_RETURN_STATUS(shape, b->GetShape(*o));
  *out_element_type = shape.element_type();
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
  ASSIGN_OR_RETURN_STATUS(executable, client->Compile(*computation, options));
  *output = executable.release();
  return nullptr;
}

status first_error(const xla_builder b) {
  MAYBE_RETURN_STATUS(b->first_error());
  return nullptr;
}

status get_current_status(const xla_builder b) {
  MAYBE_RETURN_STATUS(b->GetCurrentStatus());
  return nullptr;
}

status execute(const pjrt_loaded_executable exe, const pjrt_buffer *inputs, int ninputs, pjrt_buffer ***outputs) {
  ExecuteOptions options;
  options.strict_shape_checking = false;
  std::vector<PjRtBuffer*> input_buffer_ptrs;
  for (int i = 0; i < ninputs; ++i) {
    input_buffer_ptrs.push_back(inputs[i]);
  }
  ASSIGN_OR_RETURN_STATUS(
    results,
    exe->Execute({input_buffer_ptrs}, options));
  ASSIGN_OR_RETURN_STATUS(literal, results[0][0]->ToLiteralSync());
  pjrt_buffer** out = (pjrt_buffer**)malloc((results.size() + 1) * sizeof(pjrt_buffer*));
  for (size_t i = 0; i < results.size(); ++i) {
    auto &replica_results = results[i];
    pjrt_buffer* per_replica_outputs = (pjrt_buffer*)malloc((replica_results.size() + 1) * sizeof(pjrt_buffer));
    for (size_t j = 0; j < replica_results.size(); ++j) {
      per_replica_outputs[j] = replica_results[j].release();
    }
    per_replica_outputs[replica_results.size()] = nullptr;
    out[i] = per_replica_outputs;
  }
  out[results.size()] = nullptr;
  *outputs = out;
  return nullptr;
}

status execute_literal(const pjrt_loaded_executable exe, const literal *inputs, int ninputs, pjrt_buffer ***outputs) {
  auto client = exe->client();
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
    exe->Execute({input_buffer_ptrs}, options));
  ASSIGN_OR_RETURN_STATUS(literal, results[0][0]->ToLiteralSync());
  pjrt_buffer** out = (pjrt_buffer**)malloc((results.size() + 1) * sizeof(pjrt_buffer*));
  for (size_t i = 0; i < results.size(); ++i) {
    auto &replica_results = results[i];
    pjrt_buffer* per_replica_outputs = (pjrt_buffer*)malloc((replica_results.size() + 1) * sizeof(pjrt_buffer));
    for (size_t j = 0; j < replica_results.size(); ++j) {
      per_replica_outputs[j] = replica_results[j].release();
    }
    per_replica_outputs[replica_results.size()] = nullptr;
    out[i] = per_replica_outputs;
  }
  out[results.size()] = nullptr;
  *outputs = out;
  return nullptr;
}

literal literal_create_from_shape(int pr_type, const int64_t* dims, size_t ndims) {
  auto shape = ShapeUtil::MakeShape((PrimitiveType)pr_type, absl::Span<const int64_t>(dims, ndims));
  Literal l = Literal::CreateFromShape(shape);
  return new Literal(std::move(l));
}

int64_t literal_element_count(const literal l) {
  return l->element_count();
}

int64_t literal_size_bytes(const literal l) {
  return l->size_bytes();
}

void literal_shape(const literal l, shape *out_shape) {
  *out_shape = new Shape(l->shape());
}

int literal_element_type(const literal l) {
  return l->shape().element_type();
}

void literal_copy(const literal l, void* dst, size_t size_in_bytes) {
  std::memcpy(dst, l->untyped_data(), size_in_bytes);
}

void literal_free(literal l) {
  delete l;
}

void status_free(status s) {
  delete s;
}

char *xla_computation_name(xla_computation c) {
  return strdup(std::string(c->name()).c_str());
}

void xla_computation_free(xla_computation c) {
  delete c;
}

char *status_error_message(status s) {
  return strdup(s->error_message().c_str());
}

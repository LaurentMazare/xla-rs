#include<stdint.h>
#ifdef __cplusplus
#include <tensorflow/compiler/xla/client/xla_builder.h>
#include <tensorflow/compiler/xla/client/client_library.h>
using namespace xla;

extern "C" {
typedef XlaBuilder *xla_builder;
typedef XlaOp *xla_op;
typedef Status *status;
typedef Shape *shape;
typedef Literal *literal;
typedef GlobalData *global_data;
typedef XlaComputation *xla_computation;
#else
typedef struct _xla_builder *xla_builder;
typedef struct _xla_op *xla_op;
typedef struct _status *status;
typedef struct _shape *shape;
typedef struct _literal *literal;
typedef struct _global_data *global_data;
typedef struct _xla_computation *xla_computation;
#endif


xla_builder xla_builder_create(const char *name);
void xla_builder_free(xla_builder);

xla_op constant_r0_float(const xla_builder, float);
xla_op constant_r1_float(const xla_builder, int64_t, float);
xla_op parameter(const xla_builder, int64_t, int, int, const long int *, const char *);

// Ops
xla_op op_add(const xla_op, const xla_op);
xla_op op_sub(const xla_op, const xla_op);
xla_op op_mul(const xla_op, const xla_op);
xla_op op_div(const xla_op, const xla_op);
xla_op op_rem(const xla_op, const xla_op);
xla_op op_max(const xla_op, const xla_op);
xla_op op_min(const xla_op, const xla_op);
xla_op op_and(const xla_op, const xla_op);
xla_op op_or(const xla_op, const xla_op);
xla_op op_xor(const xla_op, const xla_op);
xla_op op_atan2(const xla_op, const xla_op);
xla_op op_pow(const xla_op, const xla_op);
xla_op op_dot(const xla_op, const xla_op);
xla_op op_not(const xla_op);
xla_op op_abs(const xla_op);
xla_op op_exp(const xla_op);
xla_op op_expm1(const xla_op);
xla_op op_floor(const xla_op);
xla_op op_ceil(const xla_op);
xla_op op_round(const xla_op);
xla_op op_log(const xla_op);
xla_op op_log1p(const xla_op);
xla_op op_logistic(const xla_op);
xla_op op_sign(const xla_op);
xla_op op_clz(const xla_op);
xla_op op_cos(const xla_op);
xla_op op_sin(const xla_op);
xla_op op_tanh(const xla_op);
xla_op op_real(const xla_op);
xla_op op_imag(const xla_op);
xla_op op_sqrt(const xla_op);
xla_op op_rsqrt(const xla_op);
xla_op op_cbrt(const xla_op);
xla_op op_is_finite(const xla_op);
xla_op op_neg(const xla_op);

int xla_op_valid(const xla_op);
void xla_op_free(xla_op);

int shape_dimensions_size(const shape);
int shape_element_type(const shape);
int64_t shape_dimensions(const shape, int);
void shape_free(shape);

status get_shape(const xla_builder, const xla_op, shape*);

status build(const xla_builder, const xla_op, xla_computation*);
status run(const xla_computation, const global_data*, int, literal *output);

// TODO: expose the xla client.
status transfer(const global_data, literal *out);
status transfer_to_server(const literal, global_data *out);

literal create_r0_f32(float);
literal create_r1_f32(const float*, int);
float literal_get_first_element_f32(const literal);
void literal_free(literal);
void global_data_free(global_data);
void xla_computation_free(xla_computation);

void status_free(status);
char *status_error_message(status);

#ifdef __cplusplus
}
#endif

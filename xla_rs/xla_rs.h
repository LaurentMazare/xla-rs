#include<stdint.h>
#ifdef __cplusplus
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <tensorflow/compiler/xla/client/client_library.h>
#include <tensorflow/compiler/xla/client/xla_builder.h>
#include <tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h>
#include <tensorflow/compiler/xla/pjrt/pjrt_client.h>
#pragma GCC diagnostic pop
using namespace xla;

extern "C" {
typedef PjRtClient *pjrt_client;
typedef PjRtLoadedExecutable *pjrt_loaded_executable;
typedef PjRtDevice *pjrt_device;
typedef PjRtBuffer *pjrt_buffer;
typedef XlaBuilder *xla_builder;
typedef XlaOp *xla_op;
typedef Status *status;
typedef Shape *shape;
typedef Literal *literal;
typedef XlaComputation *xla_computation;
#else
typedef struct _pjrt_client *pjrt_client;
typedef struct _pjrt_loaded_executable *pjrt_loaded_executable;
typedef struct _pjrt_device *pjrt_device;
typedef struct _pjrt_buffer *pjrt_buffer;
typedef struct _xla_builder *xla_builder;
typedef struct _xla_op *xla_op;
typedef struct _status *status;
typedef struct _shape *shape;
typedef struct _literal *literal;
typedef struct _xla_computation *xla_computation;
#endif

status pjrt_client_create(pjrt_client *);
void pjrt_client_free(pjrt_client);
int pjrt_client_device_count(pjrt_client);
int pjrt_client_addressable_device_count(pjrt_client);
void pjrt_client_devices(pjrt_client, pjrt_device*);
void pjrt_client_addressable_devices(pjrt_client, pjrt_device*);
char* pjrt_client_platform_name(pjrt_client);
char* pjrt_client_platform_version(pjrt_client);

void pjrt_loaded_executable_free(pjrt_loaded_executable);

int pjrt_device_id(pjrt_device);
int pjrt_device_process_index(pjrt_device);
int pjrt_device_local_hardware_id(pjrt_device);
char* pjrt_device_kind(pjrt_device);
char* pjrt_device_debug_string(pjrt_device);
char* pjrt_device_to_string(pjrt_device);

status pjrt_buffer_from_host_literal(const pjrt_client, const pjrt_device, const literal, pjrt_buffer*);
status pjrt_buffer_from_host_buffer(const pjrt_client, const pjrt_device, const void *, int, int, const int64_t*, pjrt_buffer *);
status pjrt_buffer_to_literal_sync(pjrt_buffer, literal *);
shape pjrt_buffer_on_device_shape(pjrt_buffer);
status pjrt_buffer_copy_to_device(pjrt_buffer, pjrt_device, pjrt_buffer *);
void pjrt_buffer_free(pjrt_buffer);

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
status compile(const pjrt_client, const xla_computation, pjrt_loaded_executable*);
status execute(const pjrt_loaded_executable, const pjrt_buffer *, int, pjrt_buffer ***);
status execute_literal(const pjrt_loaded_executable, const literal *, int, pjrt_buffer ***);

literal create_r0_f32(float);
literal create_r1_f32(const float*, int);
float literal_get_first_element_f32(const literal);
int64_t literal_element_count(const literal);
void literal_shape(const literal, shape*);
void literal_free(literal);
char *xla_computation_name(xla_computation);
void xla_computation_free(xla_computation);

void status_free(status);
char *status_error_message(status);

#ifdef __cplusplus
}
#endif

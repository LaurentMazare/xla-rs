#include<stdint.h>
#ifdef __cplusplus
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/client_library.h"
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
xla_op parameter(const xla_builder, int64_t, int, int, const long long int *, const char *);
xla_op add(const xla_op, const xla_op);
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

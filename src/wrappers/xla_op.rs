use super::{PrimitiveType, Shape, XlaBuilder, XlaComputation};
use crate::{c_lib, Result};

pub struct XlaOp {
    pub(super) op: c_lib::xla_op,
    pub(super) builder: XlaBuilder,
}

macro_rules! binary_op {
    ($func_name:ident, $expression:expr) => {
        pub fn $func_name(&self, op: &XlaOp) -> XlaOp {
            let op = unsafe { $expression(self.op, op.op) };
            self.wrap(op)
        }
    };
}

macro_rules! unary_op {
    ($func_name:ident, $expression:expr) => {
        pub fn $func_name(&self) -> XlaOp {
            let op = unsafe { $expression(self.op) };
            self.wrap(op)
        }
    };
}

impl XlaOp {
    pub(super) fn wrap(&self, op: c_lib::xla_op) -> Self {
        XlaOp { op, builder: self.builder.clone() }
    }

    pub fn builder(&self) -> &XlaBuilder {
        &self.builder
    }

    binary_op!(add, c_lib::op_add);
    binary_op!(sub, c_lib::op_sub);
    binary_op!(mul, c_lib::op_mul);
    binary_op!(div, c_lib::op_div);
    binary_op!(rem, c_lib::op_rem);
    binary_op!(max, c_lib::op_max);
    binary_op!(min, c_lib::op_min);
    binary_op!(and, c_lib::op_and);
    binary_op!(or, c_lib::op_or);
    binary_op!(xor, c_lib::op_xor);
    binary_op!(atan2, c_lib::op_atan2);
    binary_op!(pow, c_lib::op_pow);
    binary_op!(dot, c_lib::op_dot);

    unary_op!(not, c_lib::op_not);
    unary_op!(abs, c_lib::op_abs);
    unary_op!(exp, c_lib::op_exp);
    unary_op!(expm1, c_lib::op_expm1);
    unary_op!(floor, c_lib::op_floor);
    unary_op!(ceil, c_lib::op_ceil);
    unary_op!(round, c_lib::op_round);
    unary_op!(log, c_lib::op_log);
    unary_op!(log1p, c_lib::op_log1p);
    unary_op!(logistic, c_lib::op_logistic);
    unary_op!(sign, c_lib::op_sign);
    unary_op!(clz, c_lib::op_clz);
    unary_op!(cos, c_lib::op_cos);
    unary_op!(sin, c_lib::op_sin);
    unary_op!(tanh, c_lib::op_tanh);
    unary_op!(real, c_lib::op_real);
    unary_op!(imag, c_lib::op_imag);
    unary_op!(sqrt, c_lib::op_sqrt);
    unary_op!(rsqrt, c_lib::op_rsqrt);
    unary_op!(cbrt, c_lib::op_cbrt);
    unary_op!(is_finite, c_lib::op_is_finite);
    unary_op!(neg, c_lib::op_neg);
    unary_op!(copy, c_lib::op_copy);
    unary_op!(zeros_like, c_lib::op_zeros_like);

    pub fn reshape(&self, dims: &[i64]) -> Self {
        let op = unsafe { c_lib::op_reshape(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    pub fn broadcast(&self, dims: &[i64]) -> Self {
        let op = unsafe { c_lib::op_broadcast(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    pub fn collapse(&self, dims: &[i64]) -> Self {
        let op = unsafe { c_lib::op_collapse(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    pub fn transpose(&self, index_perm: &[i64]) -> Self {
        let op = unsafe { c_lib::op_collapse(self.op, index_perm.len(), index_perm.as_ptr()) };
        self.wrap(op)
    }

    pub fn slice_in_dim(&self, start_index: i64, stop_index: i64, stride: i64, dim: i64) -> Self {
        let op = unsafe { c_lib::op_slice_in_dim(self.op, start_index, stop_index, stride, dim) };
        self.wrap(op)
    }

    pub fn concat_in_dim(&self, args: &[&Self], dim: i64) -> Self {
        let args: Vec<_> = args.iter().map(|a| a.op).collect();
        let op = unsafe { c_lib::op_concat_in_dim(self.op, args.as_ptr(), args.len(), dim) };
        self.wrap(op)
    }

    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let op = unsafe { c_lib::op_clamp(min.op, self.op, max.op) };
        self.wrap(op)
    }

    pub fn select(&self, on_true: &Self, on_false: &Self) -> Self {
        let op = unsafe { c_lib::op_select(self.op, on_true.op, on_false.op) };
        self.wrap(op)
    }

    pub fn rng_uniform(min: &Self, max: &Self, shape: &Shape) -> Self {
        let op = unsafe {
            c_lib::op_rng_uniform(
                min.op,
                max.op,
                shape.element_type as i32,
                shape.dimensions.len() as i32,
                shape.dimensions.as_ptr(),
            )
        };
        min.wrap(op)
    }

    pub fn rng_normal(mu: &Self, sigma: &Self, shape: &Shape) -> Self {
        let op = unsafe {
            c_lib::op_rng_normal(
                mu.op,
                sigma.op,
                shape.element_type as i32,
                shape.dimensions.len() as i32,
                shape.dimensions.as_ptr(),
            )
        };
        mu.wrap(op)
    }

    pub fn convert_element_type(&self, element_type: PrimitiveType) -> Self {
        let op = unsafe { c_lib::op_convert_element_type(self.op, element_type as i32) };
        self.wrap(op)
    }

    pub fn dimension_size(&self, dim: i64) -> Self {
        let op = unsafe { c_lib::op_dimension_size(self.op, dim) };
        self.wrap(op)
    }

    pub fn reduce(&self, init_value: Self, comp: XlaComputation, dims: &[i64]) -> Self {
        let op =
            unsafe { c_lib::op_reduce(self.op, init_value.op, comp.0, dims.as_ptr(), dims.len()) };
        self.wrap(op)
    }

    pub fn element_type(&self) -> Result<PrimitiveType> {
        self.builder.get_element_type(self)
    }

    pub fn shape(&self) -> Result<Shape> {
        self.builder.get_shape(self)
    }

    // TODO: Maybe this should return [Self] rather than [Result<Self>] and encode possible errors
    // in the XlaOp?
    pub fn sum(&self, dims: &[i64]) -> Result<Self> {
        let et = self.element_type()?;
        let x = self.builder.parameter(0, et, &[], "x");
        let y = self.builder.parameter(1, et, &[], "y");
        let sum = x.add(&y).build()?;
        let init_value = self.builder.zero(et);
        Ok(self.reduce(init_value, sum, dims))
    }

    pub fn build(&self) -> Result<XlaComputation> {
        self.builder.build(self)
    }
}

impl Drop for XlaOp {
    fn drop(&mut self) {
        unsafe { c_lib::xla_op_free(self.op) }
    }
}

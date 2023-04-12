/// For details on the semantics, see https://www.tensorflow.org/xla/operation_semantics
use super::{PrimitiveType, Shape, XlaBuilder, XlaComputation};
use crate::{c_lib, Error, Result};

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

impl Clone for XlaOp {
    fn clone(&self) -> Self {
        let op = unsafe { c_lib::op_clone(self.op) };
        Self { op, builder: self.builder.clone() }
    }
}

impl XlaOp {
    pub(super) fn wrap(&self, op: c_lib::xla_op) -> Self {
        XlaOp { op, builder: self.builder.clone() }
    }

    pub fn builder(&self) -> &XlaBuilder {
        &self.builder
    }

    binary_op!(add_, c_lib::op_add);
    binary_op!(sub_, c_lib::op_sub);
    binary_op!(mul_, c_lib::op_mul);
    binary_op!(div_, c_lib::op_div);
    binary_op!(rem_, c_lib::op_rem);
    binary_op!(max, c_lib::op_max);
    binary_op!(min, c_lib::op_min);
    binary_op!(and, c_lib::op_and);
    binary_op!(or, c_lib::op_or);
    binary_op!(xor, c_lib::op_xor);
    binary_op!(atan2, c_lib::op_atan2);
    binary_op!(pow, c_lib::op_pow);
    binary_op!(dot, c_lib::op_dot);
    binary_op!(eq, c_lib::op_add);
    binary_op!(ne, c_lib::op_add);
    binary_op!(ge, c_lib::op_add);
    binary_op!(gt, c_lib::op_add);
    binary_op!(le, c_lib::op_add);
    binary_op!(lt, c_lib::op_add);

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
    unary_op!(lower_triangle, c_lib::op_lower_triangle);
    unary_op!(upper_triangle, c_lib::op_upper_triangle);
    unary_op!(copy, c_lib::op_copy);
    unary_op!(zeros_like, c_lib::op_zeros_like);

    pub fn einsum1(&self, config: &str) -> Self {
        let config = std::ffi::CString::new(config).unwrap();
        let op = unsafe { c_lib::op_einsum1(self.op, config.as_ptr()) };
        self.wrap(op)
    }

    pub fn einsum2(&self, rhs: &XlaOp, config: &str) -> Self {
        let config = std::ffi::CString::new(config).unwrap();
        let op = unsafe { c_lib::op_einsum2(self.op, rhs.op, config.as_ptr()) };
        self.wrap(op)
    }

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

    fn normalize_indexes(&self, indexes: &[i64]) -> Result<Vec<i64>> {
        let rank = self.rank()?;
        indexes
            .iter()
            .map(|&index| {
                if index >= rank as i64 {
                    Err(Error::IndexOutOfBounds { index, rank })
                } else if index >= 0 {
                    Ok(index)
                } else if index + rank as i64 >= 0 {
                    Ok(index + rank as i64)
                } else {
                    Err(Error::IndexOutOfBounds { index, rank })
                }
            })
            .collect()
    }

    fn normalize_index(&self, index: i64) -> Result<i64> {
        let rank = self.rank()?;
        if index >= rank as i64 {
            Err(Error::IndexOutOfBounds { index, rank })
        } else if index >= 0 {
            Ok(index)
        } else if index + rank as i64 >= 0 {
            Ok(index + rank as i64)
        } else {
            Err(Error::IndexOutOfBounds { index, rank })
        }
    }

    pub fn dimension_size(&self, index: i64) -> Result<Self> {
        let index = self.normalize_index(index)?;
        let op = unsafe { c_lib::op_dimension_size(self.op, index) };
        Ok(self.wrap(op))
    }

    pub fn reduce(
        &self,
        init_value: Self,
        comp: XlaComputation,
        dims: &[i64],
        keep_dims: bool,
    ) -> Result<Self> {
        let dims = self.normalize_indexes(dims)?;
        let op =
            unsafe { c_lib::op_reduce(self.op, init_value.op, comp.0, dims.as_ptr(), dims.len()) };
        let op = self.wrap(op);
        self.maybe_keep_dims(op, &dims, keep_dims)
    }

    pub fn element_type(&self) -> Result<PrimitiveType> {
        self.builder.get_element_type(self)
    }

    pub fn rank(&self) -> Result<usize> {
        self.builder.get_dimensions_size(self)
    }

    pub fn shape(&self) -> Result<Shape> {
        self.builder.get_shape(self)
    }

    pub fn gather(
        &self,
        start_indices: &XlaOp,
        offset_dims: &[i64],
        collapsed_slice_dims: &[i64],
        start_index_map: &[i64],
        set_index_vector_dim: Option<i64>,
        slice_sizes: &[i64],
    ) -> Self {
        let set_index_vector_dim = match set_index_vector_dim {
            None => std::ptr::null(),
            Some(v) => &v as *const i64,
        };
        let op = unsafe {
            c_lib::op_gather(
                self.op,
                start_indices.op,
                offset_dims.as_ptr(),
                offset_dims.len(),
                collapsed_slice_dims.as_ptr(),
                collapsed_slice_dims.len(),
                start_index_map.as_ptr(),
                start_index_map.len(),
                set_index_vector_dim,
                slice_sizes.as_ptr(),
                slice_sizes.len(),
            )
        };
        self.wrap(op)
    }

    pub fn take(&self, indices: &XlaOp, axis: i64) -> Result<Self> {
        let axis = self.normalize_index(axis)?;
        let shape = self.shape()?;
        let indices_shape = indices.shape()?;
        let index_dims = indices_shape.dimensions();
        let dims = shape.dimensions();
        let offset_dims: Vec<_> = (0..((dims.len() + index_dims.len()) as i64 - 1))
            .filter(|x| *x < axis || *x >= axis + index_dims.len() as i64)
            .collect();
        let mut slice_sizes: Vec<_> = dims.to_vec();
        slice_sizes[axis as usize] = 1;
        let mut index_dims_plus_1 = index_dims.to_vec();
        index_dims_plus_1.push(1);
        let indices = indices.reshape(&index_dims_plus_1);
        // Same as in Jax: always use the last dimension for index_vector_dim.
        let index_vector_dim = Some(index_dims.len() as i64);
        let gather =
            self.gather(&indices, &offset_dims, &[axis], &[axis], index_vector_dim, &slice_sizes);
        Ok(gather)
    }

    fn maybe_keep_dims(&self, res: XlaOp, dims: &[i64], keep_dims: bool) -> Result<XlaOp> {
        if keep_dims && !dims.is_empty() {
            let shape = self.shape()?;
            let mut dimensions = shape.dimensions().to_vec();
            for d in dims.iter() {
                dimensions[*d as usize] = 1;
            }
            Ok(res.reshape(&dimensions))
        } else {
            Ok(res)
        }
    }

    pub fn reduce_sum(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Sum");
        let et = self.element_type()?;
        let x = builder.parameter(0, et, &[], "x");
        let y = builder.parameter(1, et, &[], "y");
        let sum = x.add_(&y).build()?;
        let init_value = self.builder.zero(et);
        self.reduce(init_value, sum, dims, keep_dims)
    }

    pub fn reduce_mean(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let b = &self.builder();
        let et = self.element_type()?;
        let mut scale = b.one(PrimitiveType::S32);
        for d in dims.iter() {
            scale = scale * self.dimension_size(*d)?;
        }
        let sum = self.reduce_sum(dims, keep_dims)?;
        Ok(sum / scale.convert_element_type(et))
    }

    pub fn reduce_max(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Max");
        let et = self.element_type()?;
        let x = builder.parameter(0, et, &[], "x");
        let y = builder.parameter(1, et, &[], "y");
        let sum = x.max(&y).build()?;
        let init_value = self.builder.min_value(et);
        self.reduce(init_value, sum, dims, keep_dims)
    }

    pub fn reduce_min(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Min");
        let et = self.element_type()?;
        let x = builder.parameter(0, et, &[], "x");
        let y = builder.parameter(1, et, &[], "y");
        let sum = x.min(&y).build()?;
        let init_value = self.builder.max_value(et);
        self.reduce(init_value, sum, dims, keep_dims)
    }

    pub fn softmax(&self, dim: i64) -> Result<Self> {
        let max = self.reduce_max(&[dim], true)?;
        let unnormalized = (self - max).exp();
        let sum = unnormalized.reduce_sum(&[dim], true)?;
        Ok(unnormalized / sum)
    }

    pub fn layer_norm(&self, dim: i64, scale: &XlaOp, bias: &XlaOp) -> Result<Self> {
        let et = self.element_type().unwrap_or(PrimitiveType::F32);
        let eps = self.builder().c0(1e-5).convert_element_type(et);
        let mean = self.reduce_mean(&[dim], true)?;
        let mean2 = (self * self).reduce_mean(&[dim], true)?;
        let var = mean2 - &mean * &mean;
        let mul = (var + eps).rsqrt();
        Ok((self - mean) * mul * scale + bias)
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

macro_rules! bin_trait {
    ($trait:ident, $fn1:ident, $fn2:ident) => {
        impl<B: std::borrow::Borrow<XlaOp>> std::ops::$trait<B> for XlaOp {
            type Output = XlaOp;

            fn $fn1(self, rhs: B) -> Self::Output {
                self.$fn2(rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<XlaOp>> std::ops::$trait<B> for &XlaOp {
            type Output = XlaOp;

            fn $fn1(self, rhs: B) -> Self::Output {
                self.$fn2(rhs.borrow())
            }
        }
    };
}

bin_trait!(Add, add, add_);
bin_trait!(Sub, sub, sub_);
bin_trait!(Mul, mul, mul_);
bin_trait!(Div, div, div_);

/// For details on the semantics, see https://www.tensorflow.org/xla/operation_semantics
use super::{PrimitiveType, Shape, XlaBuilder, XlaComputation};
use crate::{c_lib, Error, Result};

pub struct XlaOp {
    pub(super) op: c_lib::xla_op,
    pub(super) builder: XlaBuilder,
}

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(&self) -> Result<$out_type> {
            let dims = self.builder.get_dims(self)?;
            if dims.len() != $cnt {
                let dims: Vec<_> = dims.iter().map(|d| *d as i64).collect();
                Err(Error::UnexpectedNumberOfDims { expected: $cnt, got: dims.len(), dims })
            } else {
                let dims = $dims(dims);
                Ok(dims)
            }
        }
    };
}

macro_rules! binary_op {
    ($func_name:ident, $expression:expr) => {
        pub fn $func_name(&self, op: &XlaOp) -> Result<Self> {
            let op = unsafe { $expression(self.op, op.op) };
            self.wrap(op)
        }
    };
}

macro_rules! unary_op {
    ($func_name:ident, $expression:expr) => {
        pub fn $func_name(&self) -> Result<Self> {
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
    pub(super) fn wrap(&self, op: c_lib::xla_op) -> Result<Self> {
        self.builder.get_current_status()?;
        Ok(XlaOp { op, builder: self.builder.clone() })
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
    binary_op!(eq, c_lib::op_eq);
    binary_op!(ne, c_lib::op_ne);
    binary_op!(ge, c_lib::op_ge);
    binary_op!(gt, c_lib::op_gt);
    binary_op!(le, c_lib::op_le);
    binary_op!(lt, c_lib::op_lt);

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

    pub fn einsum1(&self, config: &str) -> Result<Self> {
        let config = std::ffi::CString::new(config).unwrap();
        let op = unsafe { c_lib::op_einsum1(self.op, config.as_ptr()) };
        self.wrap(op)
    }

    pub fn einsum2(&self, rhs: &XlaOp, config: &str) -> Result<Self> {
        let config = std::ffi::CString::new(config).unwrap();
        let op = unsafe { c_lib::op_einsum2(self.op, rhs.op, config.as_ptr()) };
        self.wrap(op)
    }

    pub fn reshape(&self, dims: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_reshape(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    pub fn broadcast(&self, dims: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_broadcast(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    pub fn collapse(&self, dims: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_collapse(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    pub fn transpose(&self, index_perm: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_transpose(self.op, index_perm.len(), index_perm.as_ptr()) };
        self.wrap(op)
    }

    pub fn swap_dims(&self, index1: i64, index2: i64) -> Result<Self> {
        let index1 = self.normalize_index(index1)?;
        let index2 = self.normalize_index(index2)?;
        let rank = self.rank()?;
        let mut index_perm: Vec<_> = (0..rank as i64).collect();
        index_perm[index1 as usize] = index2;
        index_perm[index2 as usize] = index1;
        self.transpose(&index_perm)
    }

    pub fn slice_in_dim(
        &self,
        start_index: i64,
        stop_index: i64,
        stride: i64,
        dim: i64,
    ) -> Result<Self> {
        let dim = self.normalize_index(dim)?;
        let op = unsafe { c_lib::op_slice_in_dim(self.op, start_index, stop_index, stride, dim) };
        self.wrap(op)
    }

    pub fn slice_in_dim1(&self, start_index: i64, stop_index: i64, dim: i64) -> Result<Self> {
        self.slice_in_dim(start_index, stop_index, 1, dim)
    }

    pub fn at(&self, index_in_dim: i64, dim_index: i64) -> Result<Self> {
        let slice = self.slice_in_dim(index_in_dim, index_in_dim + 1, 1, dim_index)?;
        slice.squeeze(dim_index)
    }

    pub fn squeeze(&self, index: i64) -> Result<Self> {
        let index = self.normalize_index(index)?;
        let dims = self.dims()?;
        let mut new_dims = vec![];
        for (i, d) in dims.iter().enumerate() {
            if i as i64 != index || *d != 1 {
                new_dims.push(*d as i64)
            }
        }
        self.reshape(&new_dims)
    }

    pub fn concat_in_dim(&self, args: &[&Self], dim: i64) -> Result<Self> {
        let args: Vec<_> = args.iter().map(|a| a.op).collect();
        let op = unsafe { c_lib::op_concat_in_dim(self.op, args.as_ptr(), args.len(), dim) };
        self.wrap(op)
    }

    pub fn clamp(&self, min: &Self, max: &Self) -> Result<Self> {
        let op = unsafe { c_lib::op_clamp(min.op, self.op, max.op) };
        self.wrap(op)
    }

    pub fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let op = unsafe { c_lib::op_select(self.op, on_true.op, on_false.op) };
        self.wrap(op)
    }

    pub fn rng_uniform(min: &Self, max: &Self, shape: &Shape) -> Result<Self> {
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

    pub fn rng_normal(mu: &Self, sigma: &Self, shape: &Shape) -> Result<Self> {
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

    pub fn convert_element_type(&self, element_type: PrimitiveType) -> Result<Self> {
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

    pub fn dimensions_size(&self, index: i64) -> Result<Self> {
        let index = self.normalize_index(index)?;
        let op = unsafe { c_lib::op_dimensions_size(self.op, index) };
        self.wrap(op)
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
        let op = self.wrap(op)?;
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

    pub fn dims(&self) -> Result<Vec<usize>> {
        self.builder.get_dims(self)
    }

    extract_dims!(dim1, 1, |d: Vec<usize>| d[0], usize);
    extract_dims!(dim2, 2, |d: Vec<usize>| (d[0], d[1]), (usize, usize));
    extract_dims!(dim3, 3, |d: Vec<usize>| (d[0], d[1], d[2]), (usize, usize, usize));
    extract_dims!(dim4, 4, |d: Vec<usize>| (d[0], d[1], d[2], d[3]), (usize, usize, usize, usize));
    extract_dims!(
        dim5,
        5,
        |d: Vec<usize>| (d[0], d[1], d[2], d[3], d[4]),
        (usize, usize, usize, usize, usize)
    );

    pub fn dot_general(
        &self,
        rhs: &XlaOp,
        lhs_contracting_dims: &[i64],
        rhs_contracting_dims: &[i64],
        lhs_batch_dims: &[i64],
        rhs_batch_dims: &[i64],
    ) -> Result<Self> {
        let op = unsafe {
            c_lib::op_dot_general(
                self.op,
                rhs.op,
                lhs_contracting_dims.as_ptr(),
                lhs_contracting_dims.len(),
                rhs_contracting_dims.as_ptr(),
                rhs_contracting_dims.len(),
                lhs_batch_dims.as_ptr(),
                lhs_batch_dims.len(),
                rhs_batch_dims.as_ptr(),
                rhs_batch_dims.len(),
            )
        };
        self.wrap(op)
    }

    pub fn gather(
        &self,
        start_indices: &XlaOp,
        offset_dims: &[i64],
        collapsed_slice_dims: &[i64],
        start_index_map: &[i64],
        set_index_vector_dim: Option<i64>,
        slice_sizes: &[i64],
    ) -> Result<Self> {
        let set_index_vector_dim_ptr =
            set_index_vector_dim.as_ref().map(|p| p as *const _).unwrap_or(std::ptr::null());
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
                set_index_vector_dim_ptr,
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
        let indices = indices.reshape(&index_dims_plus_1)?;
        // Same as in Jax: always use the last dimension for index_vector_dim.
        let index_vector_dim = Some(index_dims.len() as i64);
        self.gather(&indices, &offset_dims, &[axis], &[axis], index_vector_dim, &slice_sizes)
    }

    fn maybe_keep_dims(&self, res: XlaOp, dims: &[i64], keep_dims: bool) -> Result<XlaOp> {
        if keep_dims && !dims.is_empty() {
            let shape = self.shape()?;
            let mut dimensions = shape.dimensions().to_vec();
            for d in dims.iter() {
                dimensions[*d as usize] = 1;
            }
            res.reshape(&dimensions)
        } else {
            Ok(res)
        }
    }

    pub fn reduce_sum(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Sum");
        let et = self.element_type()?;
        let x = builder.parameter(0, et, &[], "x");
        let y = builder.parameter(1, et, &[], "y");
        let sum = x.add_(&y)?.build()?;
        let init_value = self.builder.zero(et);
        self.reduce(init_value, sum, dims, keep_dims)
    }

    pub fn reduce_mean(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let b = &self.builder();
        let et = self.element_type()?;
        let mut scale = b.one(PrimitiveType::S32);
        for d in dims.iter() {
            scale = (scale * self.dimensions_size(*d)?)?;
        }
        let sum = self.reduce_sum(dims, keep_dims)?;
        sum / scale.convert_element_type(et)?
    }

    pub fn reduce_max(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Max");
        let et = self.element_type()?;
        let x = builder.parameter(0, et, &[], "x");
        let y = builder.parameter(1, et, &[], "y");
        let sum = x.max(&y)?.build()?;
        let init_value = self.builder.min_value(et);
        self.reduce(init_value, sum, dims, keep_dims)
    }

    pub fn reduce_min(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Min");
        let et = self.element_type()?;
        let x = builder.parameter(0, et, &[], "x");
        let y = builder.parameter(1, et, &[], "y");
        let sum = x.min(&y)?.build()?;
        let init_value = self.builder.max_value(et);
        self.reduce(init_value, sum, dims, keep_dims)
    }

    pub fn softmax(&self, dim: i64) -> Result<Self> {
        let max = self.reduce_max(&[dim], true)?;
        let unnormalized = (self - max)?.exp()?;
        let sum = unnormalized.reduce_sum(&[dim], true)?;
        unnormalized / sum
    }

    pub fn layer_norm(&self, dim: i64, scale: &XlaOp, bias: &XlaOp) -> Result<Self> {
        let et = self.element_type().unwrap_or(PrimitiveType::F32);
        let eps = self.builder().c0(1e-5).convert_element_type(et)?;
        let mean = self.reduce_mean(&[dim], true)?;
        let mean2 = (self * self)?.reduce_mean(&[dim], true)?;
        let var = (mean2 - (&mean * &mean)?)?;
        let mul = (var + eps)?.rsqrt()?;
        bias + ((self - mean)? * mul)? * scale
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        // Similar to the jax implementation but without the squeezing.
        // https://github.com/google/jax/blob/849e47f79ac64ccba1a762804217c00a9905025b/jax/_src/numpy/lax_numpy.py#L3028
        let lhs_shape = self.shape()?;
        let rhs_shape = self.shape()?;
        let lhs_dims = lhs_shape.dimensions();
        let rhs_dims = rhs_shape.dimensions();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        if lhs_ndims < 1 || rhs_ndims < 1 {
            Err(Error::MatMulIncorrectDims {
                lhs_dims: lhs_dims.to_vec(),
                rhs_dims: rhs_dims.to_vec(),
                msg: "empty dimension",
            })?
        }

        let rhs_is_mat = rhs_ndims > 1;
        let lhs_batch_ndims = lhs_ndims.saturating_sub(2);
        let rhs_batch_ndims = rhs_ndims.saturating_sub(2);
        let max_ndims = usize::max(lhs_batch_ndims, rhs_batch_ndims);
        let mut lhs_batch_dims = vec![];
        let mut rhs_batch_dims = vec![];
        for idx in 0..max_ndims {
            let lhs_idx = (idx + lhs_batch_ndims) as i64 - max_ndims as i64;
            let rhs_idx = (idx + rhs_batch_ndims) as i64 - max_ndims as i64;
            // Only one of lhs_idx and rhs_idx can be negative.
            if lhs_idx < 0 && rhs_idx < 0 {
                panic!("internal error: negative dim idxs {lhs_dims:?} {rhs_dims:?}")
            } else if lhs_idx < 0 && rhs_idx >= 0 {
                rhs_batch_dims.push(rhs_idx)
            } else if lhs_idx >= 0 && rhs_idx < 0 {
                lhs_batch_dims.push(lhs_idx)
            } else if lhs_dims[lhs_idx as usize] == rhs_dims[rhs_idx as usize] {
                lhs_batch_dims.push(lhs_idx);
                rhs_batch_dims.push(rhs_idx);
            } else {
                Err(Error::MatMulIncorrectDims {
                    lhs_dims: lhs_dims.to_vec(),
                    rhs_dims: rhs_dims.to_vec(),
                    msg: "incompatible batch dimensions",
                })?
            }
        }
        self.dot_general(
            rhs,
            &[lhs_ndims as i64 - 1],
            &[rhs_ndims as i64 - 1 - i64::from(rhs_is_mat)],
            &lhs_batch_dims,
            &rhs_batch_dims,
        )
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
            type Output = Result<XlaOp>;

            fn $fn1(self, rhs: B) -> Self::Output {
                (&self).$fn1(rhs)
            }
        }

        impl<B: std::borrow::Borrow<XlaOp>> std::ops::$trait<B> for &XlaOp {
            type Output = Result<XlaOp>;

            fn $fn1(self, rhs: B) -> Self::Output {
                self.$fn2(rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<XlaOp>> std::ops::$trait<Result<B>> for XlaOp {
            type Output = Result<XlaOp>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                (&self).$fn1(rhs)
            }
        }

        impl<B: std::borrow::Borrow<XlaOp>> std::ops::$trait<Result<B>> for &XlaOp {
            type Output = Result<XlaOp>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                self.$fn2(rhs?.borrow())
            }
        }
    };
}

bin_trait!(Add, add, add_);
bin_trait!(Sub, sub, sub_);
bin_trait!(Mul, mul, mul_);
bin_trait!(Div, div, div_);

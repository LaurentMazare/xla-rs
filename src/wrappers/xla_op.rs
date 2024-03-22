//! Nodes from the computation graph.
//!
//! An `XlaOp` value represents a node/operand in the computation graph, e.g. it can be the sum of two
//! other nodes, a constant value, an input parameter, etc.
//!
//! For details on the semantics, see
//! [operation_semantics](https://www.tensorflow.org/xla/operation_semantics).
use std::thread::Builder;

use super::{ArrayShape, PrimitiveType, Shape, XlaBuilder, XlaComputation};
use crate::{c_lib, ElementType, Error, Result};

pub struct XlaOp {
    pub(super) op: c_lib::xla_op,
    pub(super) builder: XlaBuilder,
}

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        #[allow(clippy::redundant_closure_call)]
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

    /// Sigmoid activation function.
    ///
    /// This computes the element-wise sigmoid.
    pub fn sigmoid(&self) -> Result<Self> {
        self.logistic()
    }

    /// SiLU activation function.
    ///
    /// This computes the element-wise SiLU activation, x.sigmoid(x).
    pub fn silu(&self) -> Result<Self> {
        self * self.logistic()
    }

    /// A node that applies the specified Einstein summation formula to this node.
    pub fn einsum1(&self, config: &str) -> Result<Self> {
        let config = std::ffi::CString::new(config).unwrap();
        let op = unsafe { c_lib::op_einsum1(self.op, config.as_ptr()) };
        self.wrap(op)
    }

    /// A node that applies the specified Einstein summation formula to this node and the other
    /// argument node.
    pub fn einsum2(&self, rhs: &XlaOp, config: &str) -> Result<Self> {
        let config = std::ffi::CString::new(config).unwrap();
        let op = unsafe { c_lib::op_einsum2(self.op, rhs.op, config.as_ptr()) };
        self.wrap(op)
    }

    /// Reshape this node to a different set of dimension sizes, the number of element between the
    /// two different shapes has to match.
    pub fn reshape(&self, dims: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_reshape(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    /// Add some broadcasting dimensions at the beginning of the current node shape.
    pub fn broadcast(&self, dims: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_broadcast(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    /// Add some broadcasting dimensions at arbitrary positions.
    ///
    /// See the [semantics](https://www.tensorflow.org/xla/operation_semantics#broadcastindim).
    pub fn broadcast_in_dim(&self, out_dims: &[i64], broadcast_dims: &[i64]) -> Result<Self> {
        let op = unsafe {
            c_lib::op_broadcast_in_dim(
                self.op,
                out_dims.len(),
                out_dims.as_ptr(),
                broadcast_dims.len(),
                broadcast_dims.as_ptr(),
            )
        };
        self.wrap(op)
    }

    /// Collapse the dimensions of this node into a single dimension, [xla
    /// documentation](https://www.tensorflow.org/xla/operation_semantics#collapse).
    pub fn collapse(&self, dims: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_collapse(self.op, dims.len(), dims.as_ptr()) };
        self.wrap(op)
    }

    /// Permute the dimension with the specified indexes.
    pub fn transpose(&self, index_perm: &[i64]) -> Result<Self> {
        let op = unsafe { c_lib::op_transpose(self.op, index_perm.len(), index_perm.as_ptr()) };
        self.wrap(op)
    }

    /// Permute two dimensions, this is a specialized version of `transpose`.
    pub fn swap_dims(&self, index1: i64, index2: i64) -> Result<Self> {
        let index1 = self.normalize_index(index1)?;
        let index2 = self.normalize_index(index2)?;
        let rank = self.rank()?;
        let mut index_perm: Vec<_> = (0..rank as i64).collect();
        index_perm[index1 as usize] = index2;
        index_perm[index2 as usize] = index1;
        self.transpose(&index_perm)
    }

    /// Create a node that has a partial view on the data of the original node. Indexes on the
    /// target dimension `dim` are restricted to the values between `start_index` (inclusive) and
    /// `stop_index` (exclusive), using the associated `stride` as a step between two values.
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

    /// A specialized version of `slice_in_dim` using a stride of one, so with all values with an
    /// index between `start_index` (inclusive) and `stop_index` (exclusive).
    pub fn slice_in_dim1(&self, start_index: i64, stop_index: i64, dim: i64) -> Result<Self> {
        self.slice_in_dim(start_index, stop_index, 1, dim)
    }

    /// A new node containing only values for index `index_in_dim` on the dimension `dim_index`.
    /// The target dimension is squeezed so the resulting node has one less dimension than the
    /// original node.
    pub fn at(&self, index_in_dim: i64, dim_index: i64) -> Result<Self> {
        let slice = self.slice_in_dim(index_in_dim, index_in_dim + 1, 1, dim_index)?;
        slice.squeeze(dim_index)
    }

    /// Squeeze the dimension as the target index, i.e. if this dimension has size one remove it
    /// for the generated node. The target dimension index can be specified as a negative value,
    /// e.g. -1 for the last dimension.
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

    /// Concat multiple nodes (together with the `self` node) along the target dimension.
    pub fn concat_in_dim<B: std::borrow::Borrow<XlaOp>>(
        &self,
        args: &[B],
        dim: i64,
    ) -> Result<Self> {
        let dim = self.normalize_index(dim)?;
        let args: Vec<_> = args.iter().map(|a| a.borrow().op).collect();
        let op = unsafe { c_lib::op_concat_in_dim(self.op, args.as_ptr(), args.len(), dim) };
        self.wrap(op)
    }

    pub fn dynamic_slice<B: std::borrow::Borrow<XlaOp>>(
        &self,
        start_indices: &[B],
        sizes: &[i64],
    ) -> Result<Self> {
        let start_indices: Vec<_> = start_indices.iter().map(|a| a.borrow().op).collect();
        let op = unsafe {
            c_lib::op_dynamic_slice(
                self.op,
                start_indices.as_ptr(),
                sizes.as_ptr(),
                start_indices.len(),
            )
        };
        self.wrap(op)
    }

    /// Index into tuples.
    pub fn get_tuple_element(&self, index: i64) -> Result<Self> {
        let op = unsafe { c_lib::op_get_tuple_element(self.op, index) };
        self.wrap(op)
    }

    /// Clamp the values in the original node to be between `min` and `max`.
    pub fn clamp(&self, min: &Self, max: &Self) -> Result<Self> {
        let op = unsafe { c_lib::op_clamp(min.op, self.op, max.op) };
        self.wrap(op)
    }

    /// Select values from the original tensor to be values from `on_true` if the associated
    /// value in `self` is true, and the values from `on_false` otherwise.
    pub fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let op = unsafe { c_lib::op_select(self.op, on_true.op, on_false.op) };
        self.wrap(op)
    }

    /// A node that when executed generates values using a random uniform distribution.
    pub fn rng_uniform(min: &Self, max: &Self, shape: &ArrayShape) -> Result<Self> {
        let dims = shape.dims();
        let op = unsafe {
            c_lib::op_rng_uniform(
                min.op,
                max.op,
                shape.primitive_type() as i32,
                dims.len() as i32,
                dims.as_ptr(),
            )
        };
        min.wrap(op)
    }

    /// A node that when executed generates values using a random normal distribution.
    pub fn rng_normal(mu: &Self, sigma: &Self, shape: &ArrayShape) -> Result<Self> {
        let dims = shape.dims();
        let op = unsafe {
            c_lib::op_rng_normal(
                mu.op,
                sigma.op,
                shape.primitive_type() as i32,
                dims.len() as i32,
                dims.as_ptr(),
            )
        };
        mu.wrap(op)
    }

    /// Create a new node by casting the elements of the original node to a new primitive type.
    pub fn convert(&self, ty: PrimitiveType) -> Result<Self> {
        let op = unsafe { c_lib::op_convert_element_type(self.op, ty as i32) };
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

    /// A node that contains the size of the dimension with the target index as a `S32` scalar
    /// value.
    pub fn dimensions_size(&self, index: i64) -> Result<Self> {
        let index = self.normalize_index(index)?;
        let op = unsafe { c_lib::op_dimensions_size(self.op, index) };
        self.wrap(op)
    }

    /// Create a node by folding a computation acress some target dimensions. If `keep_dims` is
    /// `true`, the resulting node has a dimension of size one for the target dimensions, when
    /// using `false` these dimensions are squeezed so the resulting node has a rank that is the
    /// original node rank minus the number of elements in `dims`.
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

    /// Sequentially execute `body` until `cond` fails.
    ///
    /// - `init` argument has a type `T`.
    /// - `cond` is a computation with a single argument of type `T` producing a value of type
    /// `PRED`.
    /// - `body` is a computation with a single argument of type `T` producing a value of type
    /// `T`.
    pub fn while_(cond: XlaComputation, body: XlaComputation, init: Self) -> Result<Self> {
        let op = unsafe { c_lib::op_while(cond.0, body.0, init.op) };
        init.wrap(op)
    }

    /// Execute `true_comp` if `self` is true, `false_comp` if `self` is false, and return the result.
    /// `self` has to be a scalar of type `PRED`.
    /// `true_op` is used as the single argument to `true_comp` and `false_op` as the single
    /// argument to `false_comp`.
    pub fn conditional(
        &self,
        true_op: Self,
        true_comp: XlaComputation,
        false_op: Self,
        false_comp: XlaComputation,
    ) -> Result<Self> {
        let op = unsafe {
            c_lib::op_conditional(self.op, true_op.op, true_comp.0, false_op.op, false_comp.0)
        };
        self.wrap(op)
    }

    pub fn outfeed(&self, ty: PrimitiveType, dims: &[i64], config: &str) {
        let config = std::ffi::CString::new(config).unwrap();
        unsafe {
            c_lib::outfeed(self.op, ty as i32, dims.len() as i32, dims.as_ptr(), config.as_ptr())
        }
    }

    /// The kind of elements that are computed by this operand.
    pub fn primitive_type(&self) -> Result<PrimitiveType> {
        self.builder.get_primitive_type(self)
    }

    /// The kind of elements that are computed by this operand, shortcut for `primitive_type`.
    pub fn ty(&self) -> Result<PrimitiveType> {
        self.primitive_type()
    }

    /// The number of dimensions for this node.
    pub fn rank(&self) -> Result<usize> {
        self.builder.get_dimensions_size(self)
    }

    pub fn shape(&self) -> Result<Shape> {
        self.builder.get_shape(self)
    }

    pub fn array_shape(&self) -> Result<ArrayShape> {
        ArrayShape::try_from(&self.builder.get_shape(self)?)
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

    /// General dot multiplication between two nodes, specifying the dimensions that get contracted
    /// as well as the batch dimensions.
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

    pub fn scatter(
        &self,
        indices: &XlaOp,
        updates: &XlaOp,
        update_comp: &XlaComputation,
        update_window_dims: &[i64],
        inserted_window_dims: &[i64],
        scatter_dims_to_operand_dims: &[i64],
        set_index_vector_dim: Option<i64>,
        indices_sorted: bool,
        unique_indices: bool,
    ) -> Result<Self> {
        let set_index_vector_dim_ptr =
            set_index_vector_dim.as_ref().map(|p| p as *const _).unwrap_or(std::ptr::null());
        let op = unsafe {
            c_lib::op_scatter(
                self.op,
                indices.op,
                updates.op,
                update_comp.0,
                update_window_dims.as_ptr(),
                update_window_dims.len(),
                inserted_window_dims.as_ptr(),
                inserted_window_dims.len(),
                scatter_dims_to_operand_dims.as_ptr(),
                scatter_dims_to_operand_dims.len(),
                set_index_vector_dim_ptr,
                indices_sorted,
                unique_indices,
            )
        };
        self.wrap(op)
    }

    // expects self to be a rank 1 i64 vector of indices and scatters 1s along the 2nd axis
    pub fn one_hot(&self, n_classes: i64, ty: ElementType) -> Result<Self> {
        let in_len = self.array_shape()?.dims()[0];
        let out_shape = [in_len, n_classes];

        let zero_vec: Vec<u32> = (0..out_shape.iter().product()).map(|_| 0u32).collect();
        let zeroes_r1 = self.builder.constant_r1(zero_vec.as_slice())?;
        let zeroes = zeroes_r1
                           .convert(ty.primitive_type())?
                           .reshape(out_shape.as_slice())?;

        let ones_vec: Vec<u32> = (0..in_len).map(|_| 1u32).collect();
        let ones_u32 = self.builder.constant_r1(ones_vec.as_slice())?;
        let ones = ones_u32.convert(ty.primitive_type())?.reshape(&[in_len, 1, 1])?;

        let one_hot_builder = XlaBuilder::new("one_hot");
        let _op = one_hot_builder.parameter(0, ty, &[], "operand")?;
        let up = one_hot_builder.parameter(1, ty, &[], "update")?;
        let comp = up.build()?;

        let range_vec = (0..in_len).collect::<Vec<i64>>();
        let range = self.builder.constant_r1(range_vec.as_slice())?.reshape(&[1, in_len])?;
        let indices = range.concat_in_dim(&[self.reshape(&[1, in_len])?], 0)?;

        zeroes.scatter(&indices,
                       &ones,
                       &comp,
                       &[1, 2],
                       &[],
                       &[0, 1],
                       Some(0),
                       false,
                       false)
    }

    pub fn take(&self, indices: &XlaOp, axis: i64) -> Result<Self> {
        let axis = self.normalize_index(axis)?;
        let shape = self.array_shape()?;
        let indices_shape = indices.array_shape()?;
        let index_dims = indices_shape.dims();
        let dims = shape.dims();
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

    fn maybe_keep_dims(&self, res: XlaOp, dims_to_keep: &[i64], keep_dims: bool) -> Result<XlaOp> {
        if keep_dims && !dims_to_keep.is_empty() {
            let shape = self.array_shape()?;
            let mut dims = shape.dims().to_vec();
            for d in dims_to_keep.iter() {
                dims[*d as usize] = 1;
            }
            res.reshape(&dims)
        } else {
            Ok(res)
        }
    }

    /// A node that computes the sum across the specified dimensions, e.g. if all the dimensions
    /// are passed as an argument the result is a scalar with the sum of all the elements in the
    /// original node.
    pub fn reduce_sum(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Sum");
        let ty = self.primitive_type()?.element_type()?;
        let x = builder.parameter(0, ty, &[], "x")?;
        let y = builder.parameter(1, ty, &[], "y")?;
        let sum = x.add_(&y)?.build()?;
        let init_value = self.builder.zero(ty)?;
        self.reduce(init_value, sum, dims, keep_dims)
    }

    /// A node that computes the average value across the specified dimensions.
    pub fn reduce_mean(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let b = &self.builder();
        let ty = self.primitive_type()?;
        let mut scale = b.one(crate::ElementType::S32)?;
        for d in dims.iter() {
            scale = (scale * self.dimensions_size(*d)?)?;
        }
        let sum = self.reduce_sum(dims, keep_dims)?;
        sum / scale.convert(ty)?
    }

    /// A node that computes the maximum value across the specified dimensions.
    pub fn reduce_max(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Max");
        let ty = self.primitive_type()?.element_type()?;
        let x = builder.parameter(0, ty, &[], "x")?;
        let y = builder.parameter(1, ty, &[], "y")?;
        let sum = x.max(&y)?.build()?;
        let init_value = self.builder.min_value(ty)?;
        self.reduce(init_value, sum, dims, keep_dims)
    }

    /// A node that computes the indices of maximum values across the specified dimension.
    //*
    pub fn reduce_argmax(&self, dim: i64, keep_dims: bool) -> Result<Self> {
        // we need a builder for both the while loop condition and iteration
        let cond_builder = XlaBuilder::new("Condition");
        let iter_builder = XlaBuilder::new("Iteration");

        // get self shape as slice for convenience
        let my_shape = self.array_shape()?;
        let my_shape = my_shape.dims();

        // get the data shapes of the max value accumulator and max index accumulator
        let ty = self.primitive_type()?.element_type()?;
        let mut slice_dims: Vec<i64> = Vec::new();
        for (i, d) in my_shape.iter().enumerate().rev() {
            if dim != i as i64 {
                slice_dims.push(*d);
            }
        }
        let (data_shape, index_shape) = {
            (
                Shape::Array(ArrayShape::new(slice_dims.clone(), ty)),
                Shape::Array(ArrayShape::new(slice_dims.clone(), ElementType::S64)),
            )
        };

        // build the condition i < const_len
        let const_len = cond_builder.constant_r0(my_shape[dim as usize])?;
        let cond_accum = cond_builder.parameter_s(
            0,
            &Shape::Tuple(vec![
                Shape::Array(ArrayShape::new(Vec::new(), ElementType::S64)),
                self.shape()?,
                data_shape.clone(),
                index_shape.clone(),
            ]),
            "accum",
        )?;
        let i_cond = cond_accum.get_tuple_element(0)?;
        let cond = i_cond.lt(&const_len)?.build()?;

        // declare constants and parameters for the iteration computation
        let const_zero = iter_builder.zero(ElementType::S64)?;
        let const_one = iter_builder.one(ElementType::S64)?;
        let iter_accum = iter_builder.parameter_s(
            0,
            &Shape::Tuple(vec![
                Shape::Array(ArrayShape::new(Vec::new(), ElementType::S64)),
                self.shape()?,
                data_shape,
                index_shape,
            ]),
            "accum",
        )?;

        // unpack inputs as separate arrays
        let i_iter = iter_accum.get_tuple_element(0)?;
        let me = iter_accum.get_tuple_element(1)?;
        let max_accum = iter_accum.get_tuple_element(2)?;
        let index_accum = iter_accum.get_tuple_element(3)?;

        // build bounds for dynamic slice (this seems excessive?)
        let mut starts = Vec::new();
        let mut sizes = Vec::new();
        for j in 0..my_shape.len() {
            if dim == j as i64 {
                // slice along the given dimension should start at i and be of length 1
                starts.push(&i_iter);
                sizes.push(1);
            } else {
                // slices along all other dimensions should be full size
                starts.push(&const_zero);
                sizes.push(my_shape[j]);
            }
        }


        // run the computation by comparing the current input slice to the max value accumulator
        let slice = me.dynamic_slice(&starts, &sizes)?;
        let slice_reshaped = slice.reshape(&slice_dims)?;
        let check = slice_reshaped.gt(&max_accum)?;
        let new_max = check.select(&slice_reshaped, &max_accum)?;
        let new_index = check.select(&i_iter, &index_accum)?;
        let new_i = i_iter.add_(&const_one)?;
        let argmax = iter_builder.tuple(&[new_i, me, new_max, new_index])?.build()?;

        // define initial values and build the iteration computation
        let init_index = self.builder.zero(ElementType::S64)?;
        let init_max_accum = self.slice_in_dim1(0, 1, dim)?;
        let init_max_accum = init_max_accum.reshape(&slice_dims)?;
        let n_elems = slice_dims.iter().map(|d| *d as usize).product::<usize>();
        let zero_vec = (0..n_elems).map(|_| 0i64).collect::<Vec<i64>>();
        let zero_const = self.builder.constant_r1(zero_vec.as_slice())?;
        let init_index_accum = zero_const.reshape(&slice_dims)?;
        // will this copy be fused away by XLA??
        let cpy = self.copy()?;
        let init_value =
            self.builder.tuple(&[init_index, cpy, init_max_accum, init_index_accum])?;

        // output of while loop primitive is a tuple
        let full_result = Self::while_(cond, argmax, init_value)?;

        // we only want the max indices
        let argmax = full_result.get_tuple_element(3)?;

        // maybe add a broadcasting dimension
        self.maybe_keep_dims(argmax, &[dim], keep_dims)
    }
    //*/
    /// A node that computes the indices of minimum values across the specified dimension.
    //*
    pub fn reduce_argmin(&self, dim: i64, keep_dims: bool) -> Result<Self> {
        // we need a builder for both the while loop condition and iteration
        let cond_builder = XlaBuilder::new("Condition");
        let iter_builder = XlaBuilder::new("Iteration");

        // get self shape as slice for convenience
        let my_shape = self.array_shape()?;
        let my_shape = my_shape.dims();

        // get the data shapes of the max value accumulator and max index accumulator
        let ty = self.primitive_type()?.element_type()?;
        let mut slice_dims: Vec<i64> = Vec::new();
        for (i, d) in my_shape.iter().enumerate().rev() {
            if dim != i as i64 {
                slice_dims.push(*d);
            }
        }
        let (data_shape, index_shape) = {
            (
                Shape::Array(ArrayShape::new(slice_dims.clone(), ty)),
                Shape::Array(ArrayShape::new(slice_dims.clone(), ElementType::S64)),
            )
        };

        // build the condition i < const_len
        let const_len = cond_builder.constant_r0(my_shape[dim as usize])?;
        let cond_accum = cond_builder.parameter_s(
            0,
            &Shape::Tuple(vec![
                Shape::Array(ArrayShape::new(Vec::new(), ElementType::S64)),
                self.shape()?,
                data_shape.clone(),
                index_shape.clone(),
            ]),
            "accum",
        )?;
        let i_cond = cond_accum.get_tuple_element(0)?;
        let cond = i_cond.lt(&const_len)?.build()?;

        // declare constants and parameters for the iteration computation
        let const_zero = iter_builder.zero(ElementType::S64)?;
        let const_one = iter_builder.one(ElementType::S64)?;
        let iter_accum = iter_builder.parameter_s(
            0,
            &Shape::Tuple(vec![
                Shape::Array(ArrayShape::new(Vec::new(), ElementType::S64)),
                self.shape()?,
                data_shape,
                index_shape,
            ]),
            "accum",
        )?;

        // unpack inputs as separate arrays
        let i_iter = iter_accum.get_tuple_element(0)?;
        let me = iter_accum.get_tuple_element(1)?;
        let max_accum = iter_accum.get_tuple_element(2)?;
        let index_accum = iter_accum.get_tuple_element(3)?;

        // build bounds for dynamic slice (this seems excessive?)
        let mut starts = Vec::new();
        let mut sizes = Vec::new();
        for j in (0..my_shape.len()).rev() {
            if dim == j as i64 {
                starts.push(&i_iter);
                sizes.push(1);
            } else {
                starts.push(&const_zero);
                sizes.push(my_shape[j]);
            }
        }

        // run the computation by comparing the current input slice to the max value accumulator
        let slice = me.dynamic_slice(&starts, &sizes)?;
        let slice_reshaped = slice.reshape(&slice_dims)?;
        let check = slice_reshaped.lt(&max_accum)?;
        let new_max = check.select(&slice_reshaped, &max_accum)?;
        let new_index = check.select(&i_iter, &index_accum)?;
        let new_i = i_iter.add_(&const_one)?;
        let argmax = iter_builder.tuple(&[new_i, me, new_max, new_index])?.build()?;

        // define initial values and build the iteration computation
        let init_index = self.builder.zero(ElementType::S64)?;
        let init_max_accum = self.slice_in_dim1(0, 1, dim)?;
        let init_max_accum = init_max_accum.reshape(&slice_dims)?;
        let n_elems = slice_dims.iter().map(|d| *d as usize).product::<usize>();
        let zero_vec = (0..n_elems).map(|_| 0i64).collect::<Vec<i64>>();
        let zero_const = self.builder.constant_r1(zero_vec.as_slice())?;
        let init_index_accum = zero_const.reshape(&slice_dims)?;
        // will this copy be fused away by XLA??
        let cpy = self.copy()?;
        let init_value =
            self.builder.tuple(&[init_index, cpy, init_max_accum, init_index_accum])?;

        // output of while loop primitive is a tuple
        let full_result = Self::while_(cond, argmax, init_value)?;

        // we only want the max indices
        let argmax = full_result.get_tuple_element(3)?;

        // maybe add a broadcasting dimension
        self.maybe_keep_dims(argmax, &[dim], keep_dims)
    }
    //*/
    /// A node that computes the minimum value across the specified dimensions.
    pub fn reduce_min(&self, dims: &[i64], keep_dims: bool) -> Result<Self> {
        let builder = XlaBuilder::new("Min");
        let ty = self.primitive_type()?.element_type()?;
        let x = builder.parameter(0, ty, &[], "x")?;
        let y = builder.parameter(1, ty, &[], "y")?;
        let sum = x.min(&y)?.build()?;
        let init_value = self.builder.max_value(ty)?;
        self.reduce(init_value, sum, dims, keep_dims)
    }

    pub fn softmax(&self, dim: i64) -> Result<Self> {
        let max = self.reduce_max(&[dim], true)?;
        let unnormalized = (self - max)?.exp()?;
        let sum = unnormalized.reduce_sum(&[dim], true)?;
        unnormalized / sum
    }

    /// Layer normalization, this normalizes values on the target dimension to be of zero mean and
    /// standard deviation one, and then scales the result by `scale` and adds `bias`.
    pub fn layer_norm(&self, dim: i64, scale: &XlaOp, bias: &XlaOp) -> Result<Self> {
        let ty = self.primitive_type().unwrap_or(PrimitiveType::F32);
        let eps = self.builder().c0(1e-5)?.convert(ty)?;
        let mean = self.reduce_mean(&[dim], true)?;
        let mean2 = (self * self)?.reduce_mean(&[dim], true)?;
        let var = (mean2 - (&mean * &mean)?)?;
        let mul = (var + eps)?.rsqrt()?;
        bias + ((self - mean)? * mul)? * scale
    }

    /// Matrix multiplication, this is a specialized version of `dot_general` to be used for
    /// matrix-matrix or matrix-vector multiplications.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        // Similar to the jax implementation but without the squeezing.
        // https://github.com/google/jax/blob/849matmule47f79ac64ccba1a762804217c00a9905025b/jax/_src/numpy/lax_numpy.py#L3028
        let lhs_shape = self.array_shape()?;
        let rhs_shape = self.array_shape()?;
        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();
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

    /// Generate a computation which root value is this node.
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

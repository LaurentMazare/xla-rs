use super::{ArrayElement, ElementType, PrimitiveType};
use crate::{c_lib, Error};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ArrayShape {
    ty: ElementType,
    dims: Vec<i64>,
}

impl ArrayShape {
    /// Create a new array shape.
    pub fn new<E: ArrayElement>(dims: Vec<i64>) -> Self {
        Self { ty: E::TY, dims }
    }

    /// Create a new array shape.
    pub fn new_with_type(ty: ElementType, dims: Vec<i64>) -> Self {
        Self { ty, dims }
    }

    pub fn element_type(&self) -> ElementType {
        self.ty
    }

    pub fn ty(&self) -> ElementType {
        self.ty
    }

    /// The stored primitive type.
    pub fn primitive_type(&self) -> PrimitiveType {
        self.ty.primitive_type()
    }

    /// The number of elements stored in arrays that use this shape, this is the product of sizes
    /// across each dimension.
    pub fn element_count(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product::<usize>()
    }

    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    pub fn first_dim(&self) -> Option<i64> {
        self.dims.first().copied()
    }

    pub fn last_dim(&self) -> Option<i64> {
        self.dims.last().copied()
    }
}

/// A shape specifies a primitive type as well as some array dimensions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Shape {
    Tuple(Vec<Shape>),
    Array(ArrayShape),
    Unsupported(PrimitiveType),
}

impl Shape {
    /// Create a new array shape.
    pub fn array<E: ArrayElement>(dims: Vec<i64>) -> Self {
        Self::Array(ArrayShape { ty: E::TY, dims })
    }

    /// Create a new array shape.
    pub fn array_with_type(ty: ElementType, dims: Vec<i64>) -> Self {
        Self::Array(ArrayShape { ty, dims })
    }

    /// Create a new tuple shape.
    pub fn tuple(shapes: Vec<Self>) -> Self {
        Self::Tuple(shapes)
    }

    /// The stored primitive type.
    pub fn primitive_type(&self) -> PrimitiveType {
        match self {
            Self::Tuple(_) => PrimitiveType::Tuple,
            Self::Array(a) => a.ty.primitive_type(),
            Self::Unsupported(ty) => *ty,
        }
    }

    pub fn is_tuple(&self) -> bool {
        match self {
            Self::Tuple(_) => true,
            Self::Array { .. } | Self::Unsupported(_) => false,
        }
    }

    pub fn tuple_size(&self) -> Option<usize> {
        match self {
            Self::Tuple(shapes) => Some(shapes.len()),
            Self::Array { .. } | Self::Unsupported(_) => None,
        }
    }
}

impl TryFrom<&Shape> for ArrayShape {
    type Error = Error;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        match value {
            Shape::Tuple(_) | Shape::Unsupported(_) => {
                Err(Error::NotAnArray { expected: None, got: value.clone() })
            }
            Shape::Array(a) => Ok(a.clone()),
        }
    }
}

macro_rules! extract_dims {
    ($cnt:tt, $dims:expr, $out_type:ty) => {
        impl TryFrom<&ArrayShape> for $out_type {
            type Error = Error;

            fn try_from(value: &ArrayShape) -> Result<Self, Self::Error> {
                if value.dims.len() != $cnt {
                    Err(Error::UnexpectedNumberOfDims {
                        expected: $cnt,
                        got: value.dims.len(),
                        dims: value.dims.clone(),
                    })
                } else {
                    Ok($dims(&value.dims))
                }
            }
        }

        impl TryFrom<&Shape> for $out_type {
            type Error = Error;

            fn try_from(value: &Shape) -> Result<Self, Self::Error> {
                match value {
                    Shape::Tuple(_) | Shape::Unsupported(_) => {
                        Err(Error::NotAnArray { expected: Some($cnt), got: value.clone() })
                    }
                    Shape::Array(a) => Self::try_from(a),
                }
            }
        }
    };
}

extract_dims!(1, |d: &Vec<i64>| d[0], i64);
extract_dims!(2, |d: &Vec<i64>| (d[0], d[1]), (i64, i64));
extract_dims!(3, |d: &Vec<i64>| (d[0], d[1], d[2]), (i64, i64, i64));
extract_dims!(4, |d: &Vec<i64>| (d[0], d[1], d[2], d[3]), (i64, i64, i64, i64));
extract_dims!(5, |d: &Vec<i64>| (d[0], d[1], d[2], d[3], d[4]), (i64, i64, i64, i64, i64));

pub(crate) struct CShape(c_lib::shape);

impl CShape {
    pub(crate) fn from_ptr(ptr: c_lib::shape) -> Self {
        Self(ptr)
    }

    pub(crate) fn shape(&self) -> crate::Result<Shape> {
        fn from_ptr_rec(ptr: c_lib::shape) -> crate::Result<Shape> {
            let ty = unsafe { c_lib::shape_element_type(ptr) };
            let ty = super::FromPrimitive::from_i32(ty)
                .ok_or_else(|| Error::UnexpectedElementType(ty))?;
            match ty {
                PrimitiveType::Tuple => {
                    let elem_cnt = unsafe { c_lib::shape_tuple_shapes_size(ptr) };
                    let shapes: crate::Result<Vec<_>> = (0..elem_cnt)
                        .map(|i| from_ptr_rec(unsafe { c_lib::shape_tuple_shapes(ptr, i as i32) }))
                        .collect();
                    Ok(Shape::Tuple(shapes?))
                }
                ty => match ty.element_type() {
                    Ok(ty) => {
                        let rank = unsafe { c_lib::shape_dimensions_size(ptr) };
                        let dims: Vec<_> =
                            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(ptr, i) }).collect();
                        Ok(Shape::Array(ArrayShape { ty, dims }))
                    }
                    Err(_) => Ok(Shape::Unsupported(ty)),
                },
            }
        }
        from_ptr_rec(self.0)
    }
}

impl Drop for CShape {
    fn drop(&mut self) {
        unsafe { c_lib::shape_free(self.0) };
    }
}

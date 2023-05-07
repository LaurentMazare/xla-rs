use super::{ArrayElement, ElementType, PrimitiveType};
use crate::Error;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ArrayShape {
    ty: ElementType,
    dims: Vec<i64>,
}

impl ArrayShape {
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
        }
    }

    pub fn is_tuple(&self) -> bool {
        match self {
            Self::Tuple(_) => true,
            Self::Array { .. } => false,
        }
    }

    pub fn tuple_size(&self) -> Option<usize> {
        match self {
            Self::Tuple(shapes) => Some(shapes.len()),
            Self::Array { .. } => None,
        }
    }
}

impl TryFrom<&Shape> for ArrayShape {
    type Error = Error;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        match value {
            Shape::Tuple(_) => Err(Error::NotAnArray { expected: None, got: value.clone() }),
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
                    Shape::Tuple(_) => {
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

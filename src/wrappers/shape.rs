use super::{ElementType, PrimitiveType};
use crate::Error;

/// A shape specifies a primitive type as well as some array dimensions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    pub(super) ty: PrimitiveType,
    pub(super) dimensions: Vec<i64>,
}

impl Shape {
    /// Create a new shape.
    pub fn new<E: ElementType>(dimensions: Vec<i64>) -> Shape {
        Shape { ty: E::PRIMITIVE_TYPE, dimensions }
    }

    /// Create a new shape.
    pub fn with_type(ty: PrimitiveType, dimensions: Vec<i64>) -> Shape {
        Shape { ty, dimensions }
    }

    /// The stored primitive type.
    pub fn element_type(&self) -> PrimitiveType {
        self.ty
    }

    /// The stored primitive type, shortcut for `element_type`.
    pub fn ty(&self) -> PrimitiveType {
        self.ty
    }

    /// The number of elements stored in arrays that use this shape, this is the product of sizes
    /// across each dimension.
    pub fn element_count(&self) -> usize {
        self.dimensions.iter().map(|d| *d as usize).product::<usize>()
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    pub fn first_dim(&self) -> Option<i64> {
        self.dimensions.first().copied()
    }

    pub fn last_dim(&self) -> Option<i64> {
        self.dimensions.last().copied()
    }
}

macro_rules! extract_dims {
    ($cnt:tt, $dims:expr, $out_type:ty) => {
        impl TryFrom<&Shape> for $out_type {
            type Error = Error;

            fn try_from(value: &Shape) -> Result<Self, Self::Error> {
                let dims = &value.dimensions;
                if dims.len() != $cnt {
                    Err(Error::UnexpectedNumberOfDims {
                        expected: $cnt,
                        got: dims.len(),
                        dims: dims.clone(),
                    })
                } else {
                    Ok($dims(dims))
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

use super::{ElementType, PrimitiveType};
use crate::Error;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    pub(super) element_type: PrimitiveType,
    pub(super) dimensions: Vec<i64>,
}

impl Shape {
    pub fn new<E: ElementType>(dimensions: Vec<i64>) -> Shape {
        Shape { element_type: E::PRIMITIVE_TYPE, dimensions }
    }

    pub fn with_type(element_type: PrimitiveType, dimensions: Vec<i64>) -> Shape {
        Shape { element_type, dimensions }
    }

    pub fn size(&self) -> usize {
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

impl TryFrom<&Shape> for i64 {
    type Error = Error;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        let dims = &value.dimensions;
        if dims.len() != 1 {
            Err(Error::UnexpectedNumberOfDims { expected: 1, got: dims.len(), dims: dims.clone() })
        } else {
            Ok(dims[0])
        }
    }
}

impl TryFrom<&Shape> for (i64, i64) {
    type Error = Error;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        let dims = &value.dimensions;
        if dims.len() != 2 {
            Err(Error::UnexpectedNumberOfDims { expected: 2, got: dims.len(), dims: dims.clone() })
        } else {
            Ok((dims[0], dims[1]))
        }
    }
}

impl TryFrom<&Shape> for (i64, i64, i64) {
    type Error = Error;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        let dims = &value.dimensions;
        if dims.len() != 3 {
            Err(Error::UnexpectedNumberOfDims { expected: 3, got: dims.len(), dims: dims.clone() })
        } else {
            Ok((dims[0], dims[1], dims[2]))
        }
    }
}

impl TryFrom<&Shape> for (i64, i64, i64, i64) {
    type Error = Error;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        let dims = &value.dimensions;
        if dims.len() != 4 {
            Err(Error::UnexpectedNumberOfDims { expected: 4, got: dims.len(), dims: dims.clone() })
        } else {
            Ok((dims[0], dims[1], dims[2], dims[3]))
        }
    }
}

impl TryFrom<&Shape> for (i64, i64, i64, i64, i64) {
    type Error = Error;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        let d = &value.dimensions;
        if d.len() != 5 {
            Err(Error::UnexpectedNumberOfDims { expected: 5, got: d.len(), dims: d.clone() })
        } else {
            Ok((d[0], d[1], d[2], d[3], d[4]))
        }
    }
}

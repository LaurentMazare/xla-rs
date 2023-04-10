use super::{ElementType, PrimitiveType};

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
}

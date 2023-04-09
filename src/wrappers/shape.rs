use super::{ElementType, PrimitiveType};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    pub(crate) element_type: PrimitiveType,
    pub(crate) dimensions: Vec<i64>,
}

impl Shape {
    pub fn new<E: ElementType>(dimensions: Vec<i64>) -> Shape {
        Shape { element_type: E::PRIMITIVE_TYPE, dimensions }
    }

    pub fn size(&self) -> usize {
        self.dimensions.iter().map(|d| *d as usize).product::<usize>()
    }
}

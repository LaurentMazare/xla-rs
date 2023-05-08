use anyhow::{Context, Result};
use std::collections::HashMap;

use xla::{ElementType, FromRawBytes, Literal};

#[derive(Clone)]
pub struct VarStore {
    path: Vec<String>,
    weights: std::rc::Rc<std::cell::RefCell<HashMap<String, Literal>>>,
}

impl VarStore {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let weights = xla::Literal::read_npz(path, &())?;
        let weights = weights.into_iter().collect::<HashMap<_, _>>();
        let weights = std::rc::Rc::new(std::cell::RefCell::new(weights));
        Ok(VarStore { path: vec![], weights })
    }

    pub fn len(&self) -> usize {
        self.weights.borrow().len()
    }

    pub fn take(
        &mut self,
        s: &str,
        expected_type: ElementType,
        expected_dims: &[usize],
    ) -> Result<Literal> {
        let path = format!("{}.{s}", self.path.join("."));
        let literal = self
            .weights
            .borrow_mut()
            .remove(&path)
            .with_context(|| format!("cannot find {path} in VarStore"))?;
        let shape = literal.array_shape()?;
        let element_type = shape.ty();
        let dims = shape.dims();
        if element_type != expected_type {
            anyhow::bail!(
                "unexpected element type for {}, got {:?} expected {:?}",
                path,
                element_type,
                expected_type
            )
        }
        if dims.iter().zip(expected_dims.iter()).any(|(u, v)| *u != *v as i64) {
            anyhow::bail!(
                "unexpected dims for {}, got {:?} expected {:?}",
                path,
                dims,
                expected_dims
            )
        }
        Ok(literal)
    }
}

impl<S: ToString> std::ops::Div<S> for &VarStore {
    type Output = VarStore;

    fn div(self, rhs: S) -> VarStore {
        let mut path = self.path.clone();
        path.push(rhs.to_string());
        VarStore { path, weights: self.weights.clone() }
    }
}

impl<S: ToString> std::ops::Div<S> for VarStore {
    type Output = VarStore;

    fn div(self, rhs: S) -> VarStore {
        &self / rhs
    }
}

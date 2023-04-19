use xla::{PrimitiveType, Result, XlaOp};

#[allow(dead_code)]
struct NamedVar {
    path: String,
    element_type: PrimitiveType,
    dims: Vec<usize>,
}

#[derive(Clone)]
pub struct VarBuilder {
    path: Vec<String>,
    vars: std::rc::Rc<std::cell::RefCell<Vec<NamedVar>>>,
    builder: xla::XlaBuilder,
}

impl VarBuilder {
    pub fn new(builder: &xla::XlaBuilder) -> Self {
        let vars = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        Self { builder: builder.clone(), path: vec![], vars }
    }

    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    pub fn var(&mut self, s: &str, element_type: PrimitiveType, dims: &[usize]) -> Result<XlaOp> {
        let path = format!("{}.{s}", self.path.join("."));
        let mut vars = self.vars.borrow_mut();
        let dims64 = dims.iter().map(|c| *c as i64).collect::<Vec<_>>();
        let id = vars.len();
        let parameter = self.builder.parameter(id as i64, element_type, &dims64, &path);
        vars.push(NamedVar { path, element_type, dims: dims.to_vec() });
        parameter
    }
}

impl<S: ToString> std::ops::Div<S> for &VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        let mut path = self.path.clone();
        path.push(rhs.to_string());
        VarBuilder { path, vars: self.vars.clone(), builder: self.builder.clone() }
    }
}

impl<S: ToString> std::ops::Div<S> for VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        &self / rhs
    }
}

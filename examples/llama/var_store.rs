use xla::{PrimitiveType, Result, XlaOp};

#[allow(dead_code)]
#[derive(Clone)]
struct NamedVar {
    path: String,
    et: PrimitiveType,
    dims: Vec<usize>,
    is_arg: bool,
}

#[derive(Clone)]
pub struct VarBuilder {
    path: Vec<String>,
    vars: std::rc::Rc<std::cell::RefCell<Vec<NamedVar>>>,
    builder: xla::XlaBuilder,
}

#[allow(dead_code)]
pub struct VarStore {
    vars: Vec<NamedVar>,
}

impl VarBuilder {
    pub fn new(builder: &xla::XlaBuilder) -> Self {
        let vars = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        Self { builder: builder.clone(), path: vec![], vars }
    }

    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    pub fn var_(
        &mut self,
        s: &str,
        et: PrimitiveType,
        dims: &[usize],
        is_arg: bool,
    ) -> Result<XlaOp> {
        let path = format!("{}.{s}", self.path.join("."));
        let mut vars = self.vars.borrow_mut();
        let dims64 = dims.iter().map(|c| *c as i64).collect::<Vec<_>>();
        let id = vars.len();
        let parameter = self.builder.parameter(id as i64, et, &dims64, &path);
        vars.push(NamedVar { path, et, dims: dims.to_vec(), is_arg });
        parameter
    }

    pub fn var(&mut self, s: &str, et: PrimitiveType, dims: &[usize]) -> Result<XlaOp> {
        self.var_(s, et, dims, false)
    }

    pub fn arg(&mut self, s: &str, et: PrimitiveType, dims: &[usize]) -> Result<XlaOp> {
        self.var_(s, et, dims, true)
    }

    pub fn into_store(self) -> VarStore {
        let vars = self.vars.borrow();
        VarStore { vars: vars.to_vec() }
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

impl VarStore {
    pub fn load_from_npz<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let names: Vec<_> = self
            .vars
            .iter()
            .filter_map(|n| if n.is_arg { None } else { Some(n.path.as_str()) })
            .collect();
        xla::Literal::read_npz_by_name(path, &names)?;
        Ok(())
    }
}

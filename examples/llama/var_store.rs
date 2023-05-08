use xla::{ArrayElement, ElementType, FromRawBytes, PjRtBuffer, PjRtClient, Result, XlaOp};

#[allow(dead_code)]
#[derive(Clone)]
struct NamedVar {
    path: String,
    ty: ElementType,
    dims: Vec<usize>,
    is_arg: bool,
}

#[derive(Clone)]
pub struct VarBuilder {
    path: Vec<String>,
    vars: std::rc::Rc<std::cell::RefCell<Vec<NamedVar>>>,
    builder: xla::XlaBuilder,
    default_buffer_type_for_var: ElementType,
    default_op_type_for_var: ElementType,
}

#[allow(dead_code)]
pub struct VarStore {
    vars: Vec<NamedVar>,
}

impl VarBuilder {
    pub fn new<B: ArrayElement, O: ArrayElement>(builder: &xla::XlaBuilder) -> Self {
        let vars = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        Self {
            builder: builder.clone(),
            path: vec![],
            vars,
            default_buffer_type_for_var: B::TY,
            default_op_type_for_var: O::TY,
        }
    }

    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    pub fn var_(
        &mut self,
        s: &str,
        ty: ElementType,
        dims: &[usize],
        is_arg: bool,
    ) -> Result<XlaOp> {
        let path = format!("{}.{s}", self.path.join("."));
        let mut vars = self.vars.borrow_mut();
        let dims64 = dims.iter().map(|c| *c as i64).collect::<Vec<_>>();
        let id = vars.len();
        let parameter = self.builder.parameter(id as i64, ty, &dims64, &path);
        vars.push(NamedVar { path, ty, dims: dims.to_vec(), is_arg });
        parameter
    }

    pub fn var(&mut self, s: &str, dims: &[usize]) -> Result<XlaOp> {
        let v = self.var_(s, self.default_buffer_type_for_var, dims, false)?;
        v.convert(self.default_op_type_for_var.primitive_type())
    }

    pub fn arg(&mut self, s: &str, ty: ElementType, dims: &[usize]) -> Result<XlaOp> {
        self.var_(s, ty, dims, true)
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
        VarBuilder {
            path,
            vars: self.vars.clone(),
            builder: self.builder.clone(),
            default_op_type_for_var: self.default_op_type_for_var,
            default_buffer_type_for_var: self.default_buffer_type_for_var,
        }
    }
}

impl<S: ToString> std::ops::Div<S> for VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        &self / rhs
    }
}

impl VarStore {
    pub fn arg_indexes(&self) -> Vec<usize> {
        self.vars
            .iter()
            .enumerate()
            .filter_map(|(i, n)| if n.is_arg { Some(i) } else { None })
            .collect()
    }

    pub fn load_from_npz<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        c: &PjRtClient,
    ) -> Result<Vec<PjRtBuffer>> {
        let names: Vec<_> = self
            .vars
            .iter()
            .filter_map(|n| if n.is_arg { None } else { Some(n.path.as_str()) })
            .collect();
        let mut weight_buffers = PjRtBuffer::read_npz_by_name(path, c, &names)?;
        let mut buffers = vec![];
        for var in self.vars.iter() {
            let buffer = if var.is_arg {
                let ty = var.ty;
                let element_count: usize = var.dims.iter().product();
                let element_size_in_bytes = ty.element_size_in_bytes();
                let data = vec![0u8; element_count * element_size_in_bytes];
                c.buffer_from_host_raw_bytes(ty, &data, &var.dims, None)?
            } else {
                // meh
                weight_buffers.remove(0)
            };
            buffers.push(buffer)
        }
        Ok(buffers)
    }
}

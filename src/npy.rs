// Adapted from https://github.com/LaurentMazare/tch-rs/blob/main/src/tensor/npy.rs
//! Numpy support for tensors.
//!
//! Format spec:
//! https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html
use crate::{Error, Literal, PrimitiveType, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

const NPY_MAGIC_STRING: &[u8] = b"\x93NUMPY";
const NPY_SUFFIX: &str = ".npy";

fn read_header<R: Read>(reader: &mut R) -> Result<String> {
    let mut magic_string = vec![0u8; NPY_MAGIC_STRING.len()];
    reader.read_exact(&mut magic_string)?;
    if magic_string != NPY_MAGIC_STRING {
        return Err(Error::Npy("magic string mismatch".to_string()));
    }
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;
    let header_len_len = match version[0] {
        1 => 2,
        2 => 4,
        otherwise => return Err(Error::Npy(format!("unsupported version {otherwise}"))),
    };
    let mut header_len = vec![0u8; header_len_len];
    reader.read_exact(&mut header_len)?;
    let header_len = header_len.iter().rev().fold(0_usize, |acc, &v| 256 * acc + v as usize);
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    Ok(String::from_utf8_lossy(&header).to_string())
}

#[derive(Debug, PartialEq)]
struct Header {
    descr: PrimitiveType,
    fortran_order: bool,
    shape: Vec<i64>,
}

impl Header {
    fn to_string(&self) -> Result<String> {
        let fortran_order = if self.fortran_order { "True" } else { "False" };
        let mut shape = self.shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let descr = match self.descr {
            PrimitiveType::F16 => "f2",
            PrimitiveType::F32 => "f4",
            PrimitiveType::F64 => "f8",
            PrimitiveType::S32 => "i4",
            PrimitiveType::S64 => "i8",
            PrimitiveType::S16 => "i2",
            PrimitiveType::S8 => "i1",
            PrimitiveType::U8 => "u1",
            descr => return Err(Error::Npy(format!("unsupported kind {descr:?}"))),
        };
        if !shape.is_empty() {
            shape.push(',')
        }
        Ok(format!(
            "{{'descr': '<{descr}', 'fortran_order': {fortran_order}, 'shape': ({shape}), }}"
        ))
    }

    // Hacky parser for the npy header, a typical example would be:
    // {'descr': '<f8', 'fortran_order': False, 'shape': (128,), }
    fn parse(header: &str) -> Result<Header> {
        let header =
            header.trim_matches(|c: char| c == '{' || c == '}' || c == ',' || c.is_whitespace());

        let mut parts: Vec<String> = vec![];
        let mut start_index = 0usize;
        let mut cnt_parenthesis = 0i64;
        for (index, c) in header.chars().enumerate() {
            match c {
                '(' => cnt_parenthesis += 1,
                ')' => cnt_parenthesis -= 1,
                ',' => {
                    if cnt_parenthesis == 0 {
                        parts.push(header[start_index..index].to_owned());
                        start_index = index + 1;
                    }
                }
                _ => {}
            }
        }
        parts.push(header[start_index..].to_owned());
        let mut part_map: HashMap<String, String> = HashMap::new();
        for part in parts.iter() {
            let part = part.trim();
            if !part.is_empty() {
                match part.split(':').collect::<Vec<_>>().as_slice() {
                    [key, value] => {
                        let key = key.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let value = value.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let _ = part_map.insert(key.to_owned(), value.to_owned());
                    }
                    _ => return Err(Error::Npy(format!("unable to parse header {header}"))),
                }
            }
        }
        let fortran_order = match part_map.get("fortran_order") {
            None => false,
            Some(fortran_order) => match fortran_order.as_ref() {
                "False" => false,
                "True" => true,
                _ => return Err(Error::Npy(format!("unknown fortran_order {fortran_order}"))),
            },
        };
        let descr = match part_map.get("descr") {
            None => return Err(Error::Npy("no descr in header".to_string())),
            Some(descr) => {
                if descr.is_empty() {
                    return Err(Error::Npy("empty descr".to_string()));
                }
                if descr.starts_with('>') {
                    return Err(Error::Npy(format!("little-endian descr {descr}")));
                }
                // the only supported types in tensor are:
                //     float64, float32, float16,
                //     complex64, complex128,
                //     int64, int32, int16, int8,
                //     uint8, and bool.
                match descr.trim_matches(|c: char| c == '=' || c == '<' || c == '|') {
                    "e" | "f2" => PrimitiveType::F16,
                    "f" | "f4" => PrimitiveType::F32,
                    "d" | "f8" => PrimitiveType::F64,
                    "i" | "i4" => PrimitiveType::S32,
                    "q" | "i8" => PrimitiveType::S64,
                    "h" | "i2" => PrimitiveType::S16,
                    "b" | "i1" => PrimitiveType::S8,
                    "B" | "u1" => PrimitiveType::U8,
                    "?" | "b1" => PrimitiveType::Pred,
                    "F" | "F4" => PrimitiveType::C64,
                    "D" | "F8" => PrimitiveType::C128,
                    descr => return Err(Error::Npy(format!("unrecognized descr {descr}"))),
                }
            }
        };
        let shape = match part_map.get("shape") {
            None => return Err(Error::Npy("no shape in header".to_string())),
            Some(shape) => {
                let shape = shape.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
                if shape.is_empty() {
                    vec![]
                } else {
                    shape
                        .split(',')
                        .map(|v| v.trim().parse::<i64>())
                        .collect::<std::result::Result<Vec<_>, _>>()?
                }
            }
        };
        Ok(Header { descr, fortran_order, shape })
    }
}

impl crate::Literal {
    /// Reads a npy file and return the stored tensor.
    pub fn read_npy<T: AsRef<Path>>(path: T) -> Result<Literal> {
        let mut reader = File::open(path.as_ref())?;
        let header = read_header(&mut reader)?;
        let header = Header::parse(&header)?;
        if header.fortran_order {
            return Err(Error::Npy("fortran order not supported".to_string()));
        }
        let mut data: Vec<u8> = vec![];
        reader.read_to_end(&mut data)?;
        let dims: Vec<_> = header.shape.iter().map(|v| *v as usize).collect();
        let literal = Literal::create_from_shape_and_untyped_data(header.descr, &dims, &data)?;
        Ok(literal)
    }

    /// Reads a npz file and returns some named tensors.
    pub fn read_npz<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Literal)>> {
        let zip_reader = BufReader::new(File::open(path.as_ref())?);
        let mut zip = zip::ZipArchive::new(zip_reader)?;
        let mut result = vec![];
        for i in 0..zip.len() {
            let mut reader = zip.by_index(i).unwrap();
            let name = {
                let name = reader.name();
                name.strip_suffix(NPY_SUFFIX).unwrap_or(name).to_owned()
            };
            let header = read_header(&mut reader)?;
            let header = Header::parse(&header)?;
            if header.fortran_order {
                return Err(Error::Npy("fortran order not supported".to_string()));
            }
            let mut data: Vec<u8> = vec![];
            reader.read_to_end(&mut data)?;
            let dims: Vec<_> = header.shape.iter().map(|v| *v as usize).collect();
            let literal = Literal::create_from_shape_and_untyped_data(header.descr, &dims, &data)?;
            result.push((name, literal))
        }
        Ok(result)
    }

    fn write<T: Write>(&self, f: &mut T) -> Result<()> {
        f.write_all(NPY_MAGIC_STRING)?;
        f.write_all(&[1u8, 0u8])?;
        let shape = self.shape()?;
        let header = Header {
            descr: shape.element_type(),
            fortran_order: false,
            shape: shape.dimensions().to_vec(),
        };
        let mut header = header.to_string()?;
        let pad = 16 - (NPY_MAGIC_STRING.len() + 5 + header.len()) % 16;
        for _ in 0..pad % 16 {
            header.push(' ')
        }
        header.push('\n');
        f.write_all(&[(header.len() % 256) as u8, (header.len() / 256) as u8])?;
        f.write_all(header.as_bytes())?;
        let numel = self.element_count();
        let element_type = self.element_type()?;
        let elt_size_in_bytes = element_type.element_size_in_bytes().ok_or_else(|| {
            Error::Npy(format!("unsupported element type for npy {element_type:?}"))
        })?;
        let mut content = vec![0u8; numel * elt_size_in_bytes];
        self.copy_raw_to(&mut content)?;
        f.write_all(&content)?;
        Ok(())
    }

    /// Writes a tensor in the npy format so that it can be read using python.
    pub fn write_npy<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        let mut f = File::create(path.as_ref())?;
        self.write(&mut f)
    }

    pub fn write_npz<S: AsRef<str>, T: AsRef<Literal>, P: AsRef<Path>>(
        ts: &[(S, T)],
        path: P,
    ) -> Result<()> {
        let mut zip = zip::ZipWriter::new(File::create(path.as_ref())?);
        let options =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);

        for (name, tensor) in ts.iter() {
            zip.start_file(format!("{}.npy", name.as_ref()), options)?;
            tensor.as_ref().write(&mut zip)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Header;

    #[test]
    fn parse() {
        let h = "{'descr': '<f8', 'fortran_order': False, 'shape': (128,), }";
        assert_eq!(
            Header::parse(h).unwrap(),
            Header { descr: crate::PrimitiveType::F64, fortran_order: false, shape: vec![128] }
        );
        let h = "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128), }";
        let h = Header::parse(h).unwrap();
        assert_eq!(
            h,
            Header {
                descr: crate::PrimitiveType::F32,
                fortran_order: true,
                shape: vec![256, 1, 128]
            }
        );
        assert_eq!(
            h.to_string().unwrap(),
            "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128,), }"
        );

        let h = Header { descr: crate::PrimitiveType::S64, fortran_order: false, shape: vec![] };
        assert_eq!(
            h.to_string().unwrap(),
            "{'descr': '<i8', 'fortran_order': False, 'shape': (), }"
        );
    }
}

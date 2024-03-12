#[cfg(test)]
mod tests {
    use crate::wrappers::*;

    #[test]
    fn test_dynamic_slice() {
        let builder = XlaBuilder::new("test-dynamic-slice");

        let test_tensor = builder.constant_r1(&[4, 3, 5, 9, 2, 1, 6, 7, 8]).expect("test_tensor");
        let start0 = builder.zero(ElementType::S64).expect("start0");
        let start1 = builder.one(ElementType::S64).expect("start1");
        // 4, 3, 5
        // 9, 2, 1
        // 6, 7, 8
        let test_tensor = test_tensor.reshape(&[3, 3]).expect("reshape");
        let test_argmax =
            test_tensor.dynamic_slice(&[start0, start1], &[3, 1]).expect("dynamic_slice").build().expect("build");

        let client = PjRtClient::cpu().expect("cpu");
        let executable = client.compile(&test_argmax).expect("compile");
        let result = executable.execute::<Literal>(&[]).expect("execute")[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let rust_result = result.to_vec::<i32>().expect("to_vec");

        assert_eq!(rust_result[0], 3);
        assert_eq!(rust_result[1], 2);
        assert_eq!(rust_result[2], 7);
    }

    #[test]
    fn test_argmax() {
        let builder = XlaBuilder::new("test-argmax");

        let test_tensor = builder.constant_r1(&[4, 3, 5, 9, 2, 1, 6, 7, 8]).expect("test_tensor");
        // 4, 3, 5
        // 9, 2, 1
        // 6, 7, 8
        let test_tensor = test_tensor.reshape(&[3, 3]).expect("reshape");
        let test_argmax =
            test_tensor.reduce_argmax(1, false).expect("reduce_argmax").build().expect("build");

        let client = PjRtClient::cpu().expect("cpu");
        let executable = client.compile(&test_argmax).expect("compile");
        let result = executable.execute::<Literal>(&[]).expect("execute")[0][0]
            .to_literal_sync()
            .expect("to_literal_sync");
        let rust_result = result.to_vec::<i64>().expect("to_vec");
        println!("{:?}", rust_result);
        assert_eq!(rust_result[0], 2);
        assert_eq!(rust_result[1], 0);
        assert_eq!(rust_result[2], 2);
    }
}

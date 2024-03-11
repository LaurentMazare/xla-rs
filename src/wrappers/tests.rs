#[cfg(test)]
mod tests {
    use crate::wrappers::*;

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

        assert_eq!(rust_result[0], 1);
        assert_eq!(rust_result[1], 2);
        assert_eq!(rust_result[2], 2);
    }
}

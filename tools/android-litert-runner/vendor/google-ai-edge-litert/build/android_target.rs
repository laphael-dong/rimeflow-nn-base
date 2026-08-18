#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AndroidTarget {
    pub(crate) cargo_target: &'static str,
    pub(crate) compiler_target: &'static str,
    pub(crate) cmake_abi: &'static str,
    pub(crate) bundle_arch: &'static str,
}

impl AndroidTarget {
    pub(crate) fn from_cargo_arch(arch: &str) -> Result<Self, String> {
        match arch {
            "aarch64" => Ok(Self {
                cargo_target: "aarch64-linux-android",
                compiler_target: "aarch64-linux-android26",
                cmake_abi: "arm64-v8a",
                bundle_arch: "arm64",
            }),
            "x86_64" => Ok(Self {
                cargo_target: "x86_64-linux-android",
                compiler_target: "x86_64-linux-android26",
                cmake_abi: "x86_64",
                bundle_arch: "x86_64",
            }),
            _ => Err(format!(
                "unsupported Android LiteRT target architecture: {arch}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_arm64_to_one_ndk_toolchain() {
        let target = AndroidTarget::from_cargo_arch("aarch64").unwrap();
        assert_eq!(target.cargo_target, "aarch64-linux-android");
        assert_eq!(target.compiler_target, "aarch64-linux-android26");
        assert_eq!(target.cmake_abi, "arm64-v8a");
        assert_eq!(target.bundle_arch, "arm64");
    }

    #[test]
    fn maps_x86_64_to_one_ndk_toolchain() {
        let target = AndroidTarget::from_cargo_arch("x86_64").unwrap();
        assert_eq!(target.cargo_target, "x86_64-linux-android");
        assert_eq!(target.compiler_target, "x86_64-linux-android26");
        assert_eq!(target.cmake_abi, "x86_64");
        assert_eq!(target.bundle_arch, "x86_64");
    }

    #[test]
    fn rejects_unapproved_android_architectures() {
        let error = AndroidTarget::from_cargo_arch("arm").unwrap_err();
        assert!(error.contains("unsupported Android LiteRT target architecture"));
    }
}

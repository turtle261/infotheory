macro_rules! expect_plan_ref {
    ($plan_expr:expr, $pattern:pat, $message:literal) => {
        let $pattern = $plan_expr else {
            unreachable!($message)
        };
    };
}

pub(super) use expect_plan_ref;

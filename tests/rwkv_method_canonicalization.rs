#![cfg(feature = "backend-rwkv")]

use infotheory::rwkvzip::{Compressor, MethodSpec, OnlineTrainMode, parse_method_spec};

#[test]
fn parse_cfg_named_fields_and_train_aliases() {
    let spec = parse_method_spec(
        "cfg:hidden=256,layers=1,intermediate=256,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=7,train_mode=adam,lr=0.01,stride=2",
    )
    .expect("cfg parse should succeed");

    match spec {
        MethodSpec::Online(cfg) => {
            assert_eq!(cfg.hidden, 256);
            assert_eq!(cfg.layers, 1);
            assert_eq!(cfg.intermediate, 256);
            assert_eq!(cfg.decay_rank, 8);
            assert_eq!(cfg.a_rank, 8);
            assert_eq!(cfg.v_rank, 8);
            assert_eq!(cfg.g_rank, 8);
            assert_eq!(cfg.seed, 7);
            assert!(matches!(cfg.train_mode, OnlineTrainMode::Adam));
            assert!((cfg.lr - 0.01).abs() < f32::EPSILON);
            assert_eq!(cfg.stride, 2);
        }
        other => panic!("expected online cfg spec, got {other:?}"),
    }
}

#[test]
fn parse_positional_cfg_and_canonicalize_method_string() {
    let compressor = Compressor::new_from_method("cfg:256,256,1,sgd,42,0.01,1")
        .expect("positional cfg should parse and build compressor");
    let method = compressor
        .online_method_string()
        .expect("online method should be available");
    assert_eq!(
        method,
        "cfg:hidden=256,layers=1,intermediate=256,decay_rank=32,a_rank=32,v_rank=32,g_rank=64,seed=42,train=sgd,lr=0.01,stride=1"
    );

    let reparsed = parse_method_spec(method).expect("canonical method should reparse");
    match reparsed {
        MethodSpec::Online(cfg) => {
            assert_eq!(cfg.hidden, 256);
            assert_eq!(cfg.layers, 1);
            assert_eq!(cfg.intermediate, 256);
            assert_eq!(cfg.seed, 42);
            assert!(matches!(cfg.train_mode, OnlineTrainMode::Sgd));
            assert!((cfg.lr - 0.01).abs() < f32::EPSILON);
            assert_eq!(cfg.stride, 1);
        }
        other => panic!("expected online cfg spec, got {other:?}"),
    }
}

#[test]
fn parse_rejects_unknown_cfg_key() {
    let err =
        parse_method_spec("cfg:hidden=256,unknown_key=1").expect_err("unknown cfg key should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("unknown rwkv cfg key"),
        "unexpected error message: {msg}"
    );
}

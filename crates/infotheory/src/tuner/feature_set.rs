pub(crate) fn compiled_feature_set() -> Vec<&'static str> {
    let mut features = Vec::<&'static str>::new();
    if cfg!(feature = "default-backends") {
        features.push("default-backends");
    }
    if cfg!(feature = "capability-default") {
        features.push("capability-default");
    }
    if cfg!(feature = "capability-statistical") {
        features.push("capability-statistical");
    }
    if cfg!(feature = "capability-neural") {
        features.push("capability-neural");
    }
    if cfg!(feature = "capability-archive") {
        features.push("capability-archive");
    }
    if cfg!(feature = "capability-vm") {
        features.push("capability-vm");
    }
    if cfg!(feature = "aixi") {
        features.push("aixi");
    }
    if cfg!(feature = "tuner") {
        features.push("tuner");
    }
    if cfg!(feature = "aixi-gameengine") {
        features.push("aixi-gameengine");
    }
    if cfg!(feature = "aixi-gameengine-physics") {
        features.push("aixi-gameengine-physics");
    }
    if cfg!(feature = "aixi-vm") {
        features.push("aixi-vm");
    }
    if cfg!(feature = "all-backends") {
        features.push("all-backends");
    }
    if cfg!(feature = "backend-rosa") {
        features.push("backend-rosa");
    }
    if cfg!(feature = "backend-ctw") {
        features.push("backend-ctw");
    }
    if cfg!(feature = "backend-match") {
        features.push("backend-match");
    }
    if cfg!(feature = "backend-ppmd") {
        features.push("backend-ppmd");
    }
    if cfg!(feature = "backend-sequitur") {
        features.push("backend-sequitur");
    }
    if cfg!(feature = "backend-mixture") {
        features.push("backend-mixture");
    }
    if cfg!(feature = "backend-particle") {
        features.push("backend-particle");
    }
    if cfg!(feature = "backend-calibrated") {
        features.push("backend-calibrated");
    }
    if cfg!(feature = "backend-bit-reservoir") {
        features.push("backend-bit-reservoir");
    }
    if cfg!(feature = "backend-mamba") {
        features.push("backend-mamba");
    }
    if cfg!(feature = "backend-rwkv") {
        features.push("backend-rwkv");
    }
    if cfg!(feature = "backend-zpaq") {
        features.push("backend-zpaq");
    }
    if cfg!(feature = "cli") {
        features.push("cli");
    }
    if cfg!(feature = "vm") {
        features.push("vm");
    }
    features
}

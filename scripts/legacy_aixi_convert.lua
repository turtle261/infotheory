#!/usr/bin/env luajit

local ok_cjson, cjson = pcall(require, "cjson")
if not ok_cjson then
    io.stderr:write("ERROR: lua-cjson is required (module 'cjson' not found)\n")
    os.exit(1)
end

-- Keep JSON number output stable and compact for human-facing configs.
cjson.encode_number_precision(14)

local function die(msg)
    io.stderr:write("ERROR: " .. msg .. "\n")
    os.exit(1)
end

local function trim(s)
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function lower(s)
    if s == nil then
        return ""
    end
    return string.lower(trim(tostring(s)))
end

local function dirname(path)
    local idx = path:match("^.*()/")
    if not idx then
        return "."
    end
    if idx == 1 then
        return "/"
    end
    return path:sub(1, idx - 1)
end

local function path_join(base, rel)
    if rel:sub(1, 1) == "/" then
        return rel
    end
    if base == "" or base == "." then
        return rel
    end
    if base:sub(-1) == "/" then
        return base .. rel
    end
    return base .. "/" .. rel
end

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

local function read_file(path)
    local f, err = io.open(path, "rb")
    if not f then
        die("cannot open '" .. path .. "': " .. tostring(err))
    end
    local content = f:read("*a")
    f:close()
    return content
end

local function read_stdin()
    local chunks = {}
    while true do
        local chunk = io.read(8192)
        if chunk == nil then
            break
        end
        chunks[#chunks + 1] = chunk
    end
    return table.concat(chunks)
end

local function as_int(value, field)
    if value == nil then
        return nil
    end
    local n = tonumber(value)
    if not n then
        die("field '" .. field .. "' must be numeric")
    end
    if n ~= n or n == math.huge or n == -math.huge then
        die("field '" .. field .. "' must be finite")
    end
    if n >= 0 then
        return math.floor(n)
    end
    return math.ceil(n)
end

local function as_num(value, field)
    if value == nil then
        return nil
    end
    local n = tonumber(value)
    if not n then
        die("field '" .. field .. "' must be numeric")
    end
    if n ~= n or n == math.huge or n == -math.huge then
        die("field '" .. field .. "' must be finite")
    end
    return n
end

local function as_bool(value)
    return value == true
end

local function int_with_default(value, field, default_value)
    local n = as_int(value, field)
    if n == nil then
        return default_value
    end
    return n
end

local function int_with_default_min(value, field, default_value, min_value)
    local n = int_with_default(value, field, default_value)
    if n < min_value then
        return min_value
    end
    return n
end

local function int_with_default_nonnegative(value, field, default_value)
    local n = int_with_default(value, field, default_value)
    if n < 0 then
        return 0
    end
    return n
end

local function nonnegative_num_with_default(value, field, default_value)
    local n = as_num(value, field)
    if n == nil then
        return default_value
    end
    if n < 0 then
        return 0
    end
    return n
end

local function positive_num_with_default(value, field, default_value)
    local n = as_num(value, field)
    if n == nil then
        return default_value
    end
    if n <= 0 then
        return default_value
    end
    return n
end

local function closed_unit_num_with_default(value, field, default_value)
    local n = as_num(value, field)
    if n == nil then
        return default_value
    end
    if n < 0 then
        return 0
    end
    if n > 1 then
        return 1
    end
    return n
end

local function open_unit_num_with_default(value, field, default_value)
    local n = as_num(value, field)
    if n == nil then
        return default_value
    end
    if n <= 0 then
        return default_value
    end
    if n > 1 then
        return 1
    end
    return n
end

local function next_power_of_two(n)
    local v = int_with_default_min(n, "return_bins", 1, 1)
    local p = 1
    while p < v do
        p = p * 2
    end
    return p
end

local function deep_copy(value)
    if type(value) ~= "table" then
        return value
    end
    local out = {}
    for k, v in pairs(value) do
        out[k] = deep_copy(v)
    end
    return out
end

local function normalize_rate_backend_name(name)
    local aliases = {
        ["rosaplus"] = "rosaplus",
        ["rosa"] = "rosaplus",
        ["ctw"] = "ctw",
        ["ac-ctw"] = "ctw",
        ["ctw-context-tree"] = "ctw",
        ["fac-ctw"] = "fac-ctw",
        ["facctw"] = "fac-ctw",
        ["zpaq"] = "zpaq",
        ["mamba"] = "mamba",
        ["mamba1"] = "mamba",
        ["rwkv7"] = "rwkv7",
        ["rwkv"] = "rwkv7",
        ["match"] = "match",
        ["sparse-match"] = "sparse-match",
        ["sparse_match"] = "sparse-match",
        ["sparsematch"] = "sparse-match",
        ["ppmd"] = "ppmd",
        ["ppm"] = "ppmd",
        ["sequitur"] = "sequitur",
        ["mixture"] = "mixture",
        ["mix"] = "mixture",
        ["particle"] = "particle",
        ["particles"] = "particle",
        ["calibrated"] = "calibrated",
        ["cal"] = "calibrated",
    }
    local norm = aliases[lower(name)]
    if not norm then
        die("unknown legacy rate backend '" .. tostring(name) .. "'")
    end
    return norm
end

local function signed_reward_bounds(bits)
    if bits <= 0 then
        return 0, 0
    end
    if bits >= 63 then
        return -9223372036854775808, 9223372036854775807
    end
    local max = (2 ^ (bits - 1)) - 1
    local min = -(2 ^ (bits - 1))
    return min, max
end

local function parse_observation_key_mode_str(raw)
    local s = lower(raw or "full")
    if s == "full" or s == "full-stream" or s == "stream" or s == "full_stream" then
        return "full_stream"
    end
    if s == "last" then
        return "last"
    end
    if s == "hash" or s == "stream-hash" or s == "stream_hash" then
        return "stream_hash"
    end
    return "first"
end

local function parse_observation_stream_len(value)
    return int_with_default_min(value.observation_stream_len, "observation_stream_len", 1, 1)
end

local function parse_vm_observation_stream_len(value)
    if type(value) ~= "table" then
        return 1
    end
    local stream_len = as_int(value.stream_len, "vm_observation.stream_len")
        or as_int(value.observation_stream_len, "vm_observation.observation_stream_len")
        or 1
    if stream_len < 1 then
        return 1
    end
    return stream_len
end

local function parse_observation_stream_len_for_env(root, env_name)
    if env_name == "vm" then
        if type(root.vm_observation) == "table" then
            return parse_vm_observation_stream_len(root.vm_observation)
        end
        return parse_observation_stream_len(root)
    end
    return parse_observation_stream_len(root)
end

local function parse_observation_key_mode_for_env(root, env_name)
    if env_name == "vm" then
        if type(root.vm_observation) == "table" then
            local vm_obs = root.vm_observation
            local mode = vm_obs.key_mode or vm_obs.observation_key_mode or "full"
            return parse_observation_key_mode_str(mode)
        end
        return parse_observation_key_mode_str(root.observation_key_mode or "full")
    end
    return parse_observation_key_mode_str(root.observation_key_mode or "full")
end

local function decode_hex_bytes(text, field)
    if #text % 2 ~= 0 then
        die(field .. " contains odd-length hex payload")
    end
    local out = {}
    for i = 1, #text, 2 do
        local chunk = text:sub(i, i + 1)
        local value = tonumber(chunk, 16)
        if value == nil then
            die(field .. " contains invalid hex payload")
        end
        out[#out + 1] = string.char(value)
    end
    return table.concat(out)
end

local function encode_hex_bytes(bytes)
    return (bytes:gsub('.', function(c)
        return string.format('%02x', string.byte(c))
    end))
end

local function decode_legacy_payload(text, encoding, field)
    local value = text or ""
    local enc = lower(encoding or "utf8")
    if enc == "hex" then
        return decode_hex_bytes(value, field)
    end
    return value
end

local function resolve_read_path(base_dir, path)
    if path:sub(1, 1) == "/" then
        return path
    end
    local candidate = path_join(base_dir, path)
    if file_exists(candidate) then
        return candidate
    end
    return path
end

local function as_table(value, field)
    if type(value) ~= "table" then
        die("field '" .. field .. "' must be an object")
    end
    return value
end

local function default_vm_stats_backend(root)
    local algo = lower(root.algorithm or "ctw")
    local ct_depth = int_with_default_min(root.ct_depth, "ct_depth", 20, 1)

    if algo == "ctw" or algo == "ac-ctw" or algo == "ctw-context-tree" then
        return { kind = "ctw", depth = ct_depth }
    end

    if algo == "fac-ctw" then
        return {
            kind = "fac-ctw",
            base_depth = ct_depth,
            num_percept_bits = 8,
            encoding_bits = 8,
        }
    end

    if algo == "sequitur" then
        return {
            kind = "sequitur",
            context_bytes = int_with_default_min(root.context_bytes, "context_bytes", 64, 1),
        }
    end

    if algo == "mamba" or algo == "mamba1" then
        local method = root.mamba_method
        if method ~= nil then
            return { kind = "mamba", method = method }
        end
        local model_path = root.mamba_model_path
        if not model_path then
            die("legacy default mamba backend requires mamba_model_path")
        end
        return { kind = "mamba", model_path = model_path }
    end

    if algo == "rwkv" or algo == "rwkv7" then
        local method = root.rwkv_method
        if method ~= nil then
            return { kind = "rwkv7", method = method }
        end
        local model_path = root.rwkv_model_path
        if not model_path then
            die("legacy default rwkv backend requires rwkv_model_path")
        end
        return { kind = "rwkv7", model_path = model_path }
    end

    if algo == "zpaq" then
        return {
            kind = "zpaq",
            method = tostring(root.method or "2"),
        }
    end

    if algo == "mixture" or algo == "mix" then
        local spec_path = root.mixture_spec
        if not spec_path then
            die("legacy default mixture backend requires mixture_spec")
        end
        return {
            kind = "mixture",
            spec_path = spec_path,
        }
    end

    if algo == "rosa" or algo == "rosaplus" then
        return { kind = "rosaplus" }
    end

    return { kind = "rosaplus" }
end

local function backend_from_legacy_cfg(cfg, root, base_dir, observation_bits, reward_bits)
    if cfg == nil then
        return nil
    end

    local cfg_type = type(cfg)
    local cfg_obj
    if cfg_type == "string" then
        cfg_obj = { name = cfg }
    elseif cfg_type == "table" then
        cfg_obj = cfg
    else
        die("legacy backend override must be an object or string")
    end

    local raw_name = cfg_obj.name or cfg_obj.rate_backend or cfg_obj.kind or (cfg_type == "string" and cfg or nil) or "rosaplus"
    local name = normalize_rate_backend_name(raw_name)

    if name == "rosaplus" then
        return { kind = "rosaplus" }
    end

    if name == "ctw" then
        return {
            kind = "ctw",
            depth = int_with_default_min(
                cfg_obj.ct_depth ~= nil and cfg_obj.ct_depth or cfg_obj.depth,
                "rate_backend.depth",
                32,
                1
            ),
        }
    end

    if name == "fac-ctw" then
        local encoding_bits = int_with_default_min(cfg_obj.encoding_bits, "rate_backend.encoding_bits", 8, 1)
        local percept_bits = as_int(cfg_obj.num_percept_bits, "rate_backend.num_percept_bits")
            or (observation_bits + reward_bits)
        if percept_bits < 1 then
            percept_bits = 1
        end
        return {
            kind = "fac-ctw",
            base_depth = int_with_default_min(
                cfg_obj.base_depth ~= nil and cfg_obj.base_depth or cfg_obj.ct_depth,
                "rate_backend.base_depth",
                32,
                1
            ),
            num_percept_bits = percept_bits,
            encoding_bits = encoding_bits,
        }
    end

    if name == "mamba" then
        local method = cfg_obj.method or cfg_obj.mamba_method
        if method ~= nil then
            return { kind = "mamba", method = method }
        end
        local model_path = cfg_obj.mamba_model_path or cfg_obj.model_path or root.mamba_model_path
        if not model_path then
            die("legacy mamba backend requires method/model_path")
        end
        return { kind = "mamba", model_path = model_path }
    end

    if name == "rwkv7" then
        local method = cfg_obj.method or cfg_obj.rwkv_method
        if method ~= nil then
            return { kind = "rwkv7", method = method }
        end
        local model_path = cfg_obj.rwkv_model_path or cfg_obj.model_path or root.rwkv_model_path
        if not model_path then
            die("legacy rwkv backend requires method/model_path")
        end
        return { kind = "rwkv7", model_path = model_path }
    end

    if name == "zpaq" then
        return {
            kind = "zpaq",
            method = tostring(cfg_obj.method or cfg_obj.zpaq_method or root.method or "2"),
        }
    end

    if name == "match" then
        local min_len = int_with_default_min(cfg_obj.min_len, "rate_backend.min_len", 4, 1)
        local max_len = int_with_default_min(cfg_obj.max_len, "rate_backend.max_len", 255, 1)
        if max_len < min_len then
            max_len = min_len
        end
        return {
            kind = "match",
            hash_bits = int_with_default_min(cfg_obj.hash_bits, "rate_backend.hash_bits", 20, 1),
            min_len = min_len,
            max_len = max_len,
            base_mix = nonnegative_num_with_default(cfg_obj.base_mix, "rate_backend.base_mix", 0.02),
            confidence_scale = positive_num_with_default(cfg_obj.confidence_scale, "rate_backend.confidence_scale", 1.0),
        }
    end

    if name == "sparse-match" then
        local min_len = int_with_default_min(cfg_obj.min_len, "rate_backend.min_len", 3, 1)
        local max_len = int_with_default_min(cfg_obj.max_len, "rate_backend.max_len", 64, 1)
        if max_len < min_len then
            max_len = min_len
        end
        return {
            kind = "sparse-match",
            hash_bits = int_with_default_min(cfg_obj.hash_bits, "rate_backend.hash_bits", 19, 1),
            min_len = min_len,
            max_len = max_len,
            gap_min = int_with_default_min(cfg_obj.gap_min, "rate_backend.gap_min", 1, 0),
            gap_max = int_with_default_min(cfg_obj.gap_max, "rate_backend.gap_max", 2, 0),
            base_mix = nonnegative_num_with_default(cfg_obj.base_mix, "rate_backend.base_mix", 0.05),
            confidence_scale = positive_num_with_default(cfg_obj.confidence_scale, "rate_backend.confidence_scale", 1.0),
        }
    end

    if name == "ppmd" then
        return {
            kind = "ppmd",
            order = int_with_default_min(cfg_obj.order, "rate_backend.order", 10, 1),
            memory_mb = int_with_default_min(cfg_obj.memory_mb, "rate_backend.memory_mb", 64, 1),
        }
    end

    if name == "sequitur" then
        return {
            kind = "sequitur",
            context_bytes = int_with_default_min(cfg_obj.context_bytes, "rate_backend.context_bytes", 64, 1),
        }
    end

    if name == "mixture" then
        if type(cfg_obj.spec) == "table" then
            return { kind = "mixture", spec = deep_copy(cfg_obj.spec) }
        end
        local spec_path = cfg_obj.mixture_spec or cfg_obj.spec_path or cfg_obj.path
        if type(cfg_obj.spec) == "string" and spec_path == nil then
            spec_path = cfg_obj.spec
        end
        if spec_path == nil then
            spec_path = root.mixture_spec
        end
        if not spec_path then
            die("legacy mixture backend requires inline spec or mixture_spec/spec_path")
        end
        return { kind = "mixture", spec_path = spec_path }
    end

    if name == "particle" then
        if type(cfg_obj.spec) == "table" then
            return { kind = "particle", spec = deep_copy(cfg_obj.spec) }
        end
        local spec_path = cfg_obj.particle_spec or cfg_obj.spec_path or cfg_obj.path
        if type(cfg_obj.spec) == "string" and spec_path == nil then
            spec_path = cfg_obj.spec
        end
        if spec_path == nil then
            spec_path = root.particle_spec
        end
        if spec_path then
            return { kind = "particle", spec_path = spec_path }
        end
        local inline = deep_copy(cfg_obj)
        inline.kind = "particle"
        inline.name = nil
        inline.rate_backend = nil
        return inline
    end

    if name == "calibrated" then
        if type(cfg_obj.spec) == "table" then
            return { kind = "calibrated", spec = deep_copy(cfg_obj.spec) }
        end
        local spec_path = cfg_obj.calibrated_spec or cfg_obj.spec_path or cfg_obj.path
        if type(cfg_obj.spec) == "string" and spec_path == nil then
            spec_path = cfg_obj.spec
        end
        if spec_path == nil then
            spec_path = root.calibrated_spec
        end
        if spec_path then
            return { kind = "calibrated", spec_path = spec_path }
        end
        local inline = deep_copy(cfg_obj)
        inline.kind = "calibrated"
        inline.name = nil
        inline.rate_backend = nil
        return inline
    end

    die("unsupported legacy backend kind '" .. tostring(raw_name) .. "'")
end

local function parse_vm_observation_policy(mode)
    local m = lower(mode or "guest")
    if m == "raw" or m == "raw-bytes" or m == "bytes" or m == "stream" then
        return "raw_output"
    end
    if m == "hash" or m == "output-hash" then
        return "output_hash"
    end
    if m == "shared-memory" or m == "shared_mem" or m == "shared" then
        return "shared_memory"
    end
    return "from_guest"
end

local function parse_vm_observation_stream_mode(mode)
    local m = lower(mode or "pad-truncate")
    if m == "pad" then
        return "pad"
    end
    if m == "truncate" then
        return "truncate"
    end
    return "pad_truncate"
end

local function parse_payload_encoding(mode)
    local m = lower(mode or "utf8")
    if m == "hex" then
        return "hex"
    end
    return "utf8"
end

local function parse_wire_encoding(mode)
    local m = lower(mode or "hex")
    if m == "utf8" or m == "text" then
        return "utf8"
    end
    if m == "hex" then
        return "hex"
    end
    return "hex"
end

local function parse_vm_fuzz_mutator(name)
    local n = lower(name)
    if n == "flip_bit" or n == "flipbit" then
        return "flip_bit"
    end
    if n == "flip_byte" or n == "flipbyte" then
        return "flip_byte"
    end
    if n == "insert" or n == "insert_byte" or n == "insertbyte" then
        return "insert_byte"
    end
    if n == "delete" or n == "delete_byte" or n == "deletebyte" then
        return "delete_byte"
    end
    if n == "splice" or n == "splice_seed" or n == "splice-seed" then
        return "splice_seed"
    end
    if n == "reset" or n == "reset_seed" or n == "reset-seed" then
        return "reset_seed"
    end
    if n == "havoc" then
        return "havoc"
    end
    return nil
end

local function make_asset_registry()
    local assets = {}
    local by_path = {}
    local used = {}

    local function sanitize(name)
        local out = tostring(name or "asset")
        out = out:gsub("[^%w_%-]+", "_")
        out = out:gsub("_+", "_")
        out = out:gsub("^_+", "")
        out = out:gsub("_+$", "")
        if out == "" then
            out = "asset"
        end
        return out
    end

    local function add(path, hint)
        if path == nil then
            return nil
        end
        local key = tostring(path)
        local existing = by_path[key]
        if existing then
            return existing
        end
        local base = sanitize(hint)
        local id = base
        local i = 2
        while used[id] do
            id = base .. "_" .. i
            i = i + 1
        end
        used[id] = true
        by_path[key] = id
        assets[#assets + 1] = {
            id = id,
            path = key,
        }
        return id
    end

    local function list()
        return assets
    end

    return {
        add = add,
        list = list,
    }
end

local function convert_legacy_vm_action_source(raw, base_dir)
    local source = type(raw) == "table" and raw or {}
    local mode = lower(source.mode or "literal")

    if mode == "fuzz" then
        local fuzz = type(source.fuzz) == "table" and source.fuzz or source
        local seed_encoding = parse_payload_encoding(fuzz.seed_encoding)
        local dict_encoding = parse_payload_encoding(fuzz.dict_encoding)

        local seeds = {}
        if type(fuzz.seed_paths) == "table" then
            for idx, path in ipairs(fuzz.seed_paths) do
                if type(path) == "string" then
                    local resolved = resolve_read_path(base_dir, path)
                    local bytes = read_file(resolved)
                    seeds[#seeds + 1] = encode_hex_bytes(bytes)
                else
                    die("vm_actions.fuzz.seed_paths[" .. idx .. "] must be a string")
                end
            end
        end
        if type(fuzz.seed_inputs) == "table" then
            for idx, text in ipairs(fuzz.seed_inputs) do
                if type(text) ~= "string" then
                    die("vm_actions.fuzz.seed_inputs[" .. idx .. "] must be a string")
                end
                local bytes = decode_legacy_payload(text, seed_encoding, "vm_actions.fuzz.seed_inputs")
                seeds[#seeds + 1] = encode_hex_bytes(bytes)
            end
        end

        local mutators = {}
        if type(fuzz.mutators) == "table" then
            for _, name in ipairs(fuzz.mutators) do
                if type(name) == "string" then
                    local canonical = parse_vm_fuzz_mutator(name)
                    if canonical ~= nil then
                        mutators[#mutators + 1] = canonical
                    end
                end
            end
        end

        local dictionary = {}
        if type(fuzz.dictionary) == "table" then
            for idx, text in ipairs(fuzz.dictionary) do
                if type(text) ~= "string" then
                    die("vm_actions.fuzz.dictionary[" .. idx .. "] must be a string")
                end
                local bytes = decode_legacy_payload(text, dict_encoding, "vm_actions.fuzz.dictionary")
                dictionary[#dictionary + 1] = encode_hex_bytes(bytes)
            end
        end

        if #mutators == 0 then
            mutators[1] = "havoc"
        end

        local min_len = int_with_default_min(fuzz.min_len, "vm_actions.fuzz.min_len", 1, 1)
        local max_len = int_with_default_min(fuzz.max_len, "vm_actions.fuzz.max_len", 4096, 1)
        if max_len < min_len then
            max_len = min_len
        end
        local rng_seed = int_with_default_nonnegative(fuzz.rng_seed, "vm_actions.fuzz.rng_seed", 0)

        if #seeds == 0 then
            seeds[1] = ""
        end

        return {
            kind = "fuzz",
            seeds = seeds,
            encoding = "hex",
            mutators = mutators,
            min_len = min_len,
            max_len = max_len,
            dictionary = dictionary,
            rng_seed = rng_seed,
        }
    end

    local actions = {}
    local raw_actions = source.actions
    if type(raw_actions) == "table" then
        for idx, entry in ipairs(raw_actions) do
            if type(entry) == "string" then
                local bytes = decode_legacy_payload(entry, "utf8", "vm_actions.actions")
                actions[#actions + 1] = { payload = encode_hex_bytes(bytes) }
            elseif type(entry) == "table" then
                local payload = entry.payload
                if payload ~= nil and type(payload) ~= "string" then
                    die("vm_actions.actions[" .. idx .. "].payload must be a string")
                end
                local bytes = decode_legacy_payload(payload or "", entry.encoding, "vm_actions.actions")
                local action = { payload = encode_hex_bytes(bytes) }
                if entry.name ~= nil then
                    action.name = tostring(entry.name)
                end
                actions[#actions + 1] = action
            else
                die("vm_actions.actions[" .. idx .. "] must be a string or object")
            end
        end
    end

    if #actions == 0 then
        actions[1] = { payload = "" }
    end

    return {
        kind = "literal",
        encoding = "hex",
        actions = actions,
    }
end

local function convert_legacy_vm_environment(root, base_dir, assets)
    local vm = as_table(root.vm_config, "vm_config")

    local observation_bits = int_with_default_min(root.observation_bits, "observation_bits", 16, 1)
    local reward_bits = int_with_default_min(root.reward_bits, "reward_bits", 8, 1)
    local agent_horizon = int_with_default_min(root.agent_horizon, "agent_horizon", 3, 1)

    local firecracker_path = vm.firecracker_config or vm.config or root.firecracker_config
    if firecracker_path == nil then
        die("legacy vm config requires vm_config.firecracker_config")
    end
    if type(firecracker_path) ~= "string" then
        die("vm_config.firecracker_config must be a string")
    end
    local firecracker_asset = assets.add(firecracker_path, "firecracker")

    local vm_observation = type(vm.observation) == "table" and vm.observation
        or (type(root.vm_observation) == "table" and root.vm_observation)
        or nil

    local vm_reward = type(vm.reward) == "table" and vm.reward
        or (type(root.vm_reward) == "table" and root.vm_reward)
        or {}

    local vm_actions = type(vm.actions) == "table" and vm.actions
        or (type(root.vm_actions) == "table" and root.vm_actions)
        or {}

    local vm_trace = type(vm.trace) == "table" and vm.trace
        or (type(root.vm_trace) == "table" and root.vm_trace)
        or nil

    local vm_filter = type(vm.filter) == "table" and vm.filter
        or (type(root.vm_filter) == "table" and root.vm_filter)
        or nil

    local protocol = type(vm.protocol) == "table" and vm.protocol
        or (type(root.vm_protocol) == "table" and root.vm_protocol)
        or {}

    local action_source = convert_legacy_vm_action_source(vm_actions, base_dir)
    local action_count
    if action_source.kind == "literal" then
        action_count = #action_source.actions
    else
        action_count = #action_source.mutators
    end

    local reward_policy
    do
        local reward_mode = lower(vm_reward.mode or "guest")
        if reward_mode == "pattern" then
            local pattern = vm_reward.pattern
            if type(pattern) ~= "string" then
                die("vm_reward.pattern is required when vm_reward.mode=pattern")
            end
            reward_policy = {
                kind = "pattern",
                pattern = pattern,
                base_reward = as_int(vm_reward.base_reward, "vm_reward.base_reward") or 0,
                bonus_reward = as_int(vm_reward.bonus_reward, "vm_reward.bonus_reward") or 10,
            }
        else
            reward_policy = { kind = "from_guest" }
        end
    end

    local reward_shaping
    do
        local shaping = nil
        if type(vm.reward_shaping) == "table" then
            shaping = vm.reward_shaping
        elseif type(root.vm_reward_shaping) == "table" then
            shaping = root.vm_reward_shaping
        elseif type(vm.reward) == "table" and type(vm.reward.shaping) == "table" then
            shaping = vm.reward.shaping
        end

        if shaping ~= nil then
            local mode = lower(shaping.mode or "none")
            if mode == "entropy-reduction" or mode == "entropy_reduction" then
                local baseline_path = shaping.baseline_path
                if type(baseline_path) ~= "string" then
                    die("vm_reward_shaping.baseline_path is required in entropy-reduction mode")
                end
                reward_shaping = {
                    kind = "entropy_reduction",
                    baseline_asset = assets.add(baseline_path, "baseline"),
                    max_order = as_int(shaping.max_order, "vm_reward_shaping.max_order") or 8,
                    scale = as_num(shaping.scale, "vm_reward_shaping.scale") or 10.0,
                    crash_bonus = as_int(shaping.crash_bonus, "vm_reward_shaping.crash_bonus"),
                    timeout_bonus = as_int(shaping.timeout_bonus, "vm_reward_shaping.timeout_bonus"),
                }
            elseif mode == "trace-entropy" or mode == "trace_entropy" then
                reward_shaping = {
                    kind = "trace_entropy",
                    max_order = as_int(shaping.max_order, "vm_reward_shaping.max_order") or 8,
                    scale = as_num(shaping.scale, "vm_reward_shaping.scale") or 1.0,
                    normalize = as_bool(shaping.normalize),
                }
            end
        end
    end

    local action_filter
    if vm_filter ~= nil then
        local novelty_prior_asset
        if type(vm_filter.novelty_prior_path) == "string" then
            novelty_prior_asset = assets.add(vm_filter.novelty_prior_path, "novelty_prior")
        end
        local reject_reward = as_int(vm_filter.reject_reward, "vm_filter.reject_reward")
        if reject_reward == nil then
            reject_reward = -(as_int(vm.step_cost, "vm_config.step_cost") or 1)
        end
        action_filter = {
            min_entropy = as_num(vm_filter.min_entropy, "vm_filter.min_entropy"),
            max_entropy = as_num(vm_filter.max_entropy, "vm_filter.max_entropy"),
            min_intrinsic_dependence = as_num(vm_filter.min_intrinsic_dependence, "vm_filter.min_intrinsic_dependence"),
            min_novelty = as_num(vm_filter.min_novelty, "vm_filter.min_novelty"),
            novelty_prior_asset = novelty_prior_asset,
            max_order = as_int(vm_filter.max_order, "vm_filter.max_order") or 8,
            reject_reward = reject_reward,
        }
    end

    local trace
    if vm_trace ~= nil then
        local shared_region_name = vm_trace.shared_region_name
            or vm_trace.shared_region
            or vm_trace.name
            or ((lower(vm_trace.mode) == "shared-memory") and "trace" or nil)
            or "trace"
        trace = {
            shared_region_name = shared_region_name,
            max_bytes = as_int(vm_trace.max_bytes, "vm_trace.max_bytes") or 1000000,
            reset_on_episode = as_bool(vm_trace.reset_on_episode),
        }
    end

    local stats_backend = backend_from_legacy_cfg(
        vm.stats_backend or root.vm_stats_backend,
        root,
        base_dir,
        observation_bits,
        reward_bits
    )
    if stats_backend == nil then
        stats_backend = default_vm_stats_backend(root)
    end

    local environment = {
        kind = "nyx_vm",
        firecracker_config_asset = firecracker_asset,
        instance_id = tostring(vm.instance_id or "aixi-nyx"),
        shared_region_name = tostring(vm.shared_region_name or "shared"),
        shared_region_size = int_with_default_min(vm.shared_region_size, "vm_config.shared_region_size", 4096, 1),
        shared_memory_policy = (function()
            local policy = lower(vm.shared_memory_policy or root.shared_memory_policy or "snapshot")
            if policy == "preserve" or policy == "keep" then
                return "preserve"
            end
            return "snapshot"
        end)(),
        step_timeout_ms = int_with_default_min(vm.step_timeout_ms, "vm_config.step_timeout_ms", 100, 1),
        boot_timeout_ms = int_with_default_min(vm.boot_timeout_ms, "vm_config.boot_timeout_ms", 30000, 1),
        episode_steps = int_with_default_min(vm.episode_steps, "vm_config.episode_steps", agent_horizon, 1),
        step_cost = as_int(vm.step_cost, "vm_config.step_cost") or 1,
        observation_policy = parse_vm_observation_policy(vm_observation and vm_observation.mode or nil),
        observation_bits = observation_bits,
        observation_stream_len = parse_vm_observation_stream_len(vm_observation),
        observation_stream_mode = parse_vm_observation_stream_mode(vm_observation and vm_observation.stream_mode or nil),
        observation_pad_byte = as_int(vm_observation and vm_observation.pad_byte or nil, "vm_observation.pad_byte") or 0,
        reward_bits = reward_bits,
        reward_policy = reward_policy,
        reward_shaping = reward_shaping,
        action_source = action_source,
        action_filter = action_filter,
        protocol = {
            action_prefix = tostring(protocol.action_prefix or "ACT "),
            action_suffix = tostring(protocol.action_suffix or "\n"),
            obs_prefix = tostring(protocol.obs_prefix or "OBS "),
            rew_prefix = tostring(protocol.rew_prefix or "REW "),
            done_prefix = tostring(protocol.done_prefix or "DONE "),
            data_prefix = tostring(protocol.data_prefix or "DATA "),
            wire_encoding = parse_wire_encoding(protocol.wire_encoding),
        },
        stats_backend = stats_backend,
        trace = trace,
        debug_mode = as_bool(vm.verbose) or as_bool(vm.debug),
        crash_log = (type(vm.crash_log) == "string") and vm.crash_log or nil,
    }

    return environment, {
        observation_bits = observation_bits,
        reward_bits = reward_bits,
        action_count = action_count,
    }
end

local BUILTIN_ENV = {
    ["coin-flip"] = "coin_flip",
    ["coin_flip"] = "coin_flip",
    ["coinflip"] = "coin_flip",
    ["ctw-test"] = "ctw_test",
    ["ctw_test"] = "ctw_test",
    ["tictactoe"] = "tic_tac_toe",
    ["tic-tac-toe"] = "tic_tac_toe",
    ["tic_tac_toe"] = "tic_tac_toe",
    ["extended-tiger"] = "extended_tiger",
    ["extended_tiger"] = "extended_tiger",
    ["biased-rock-paper-scissor"] = "biased_rock_paper_scissor",
    ["biased-rock-paper-scissors"] = "biased_rock_paper_scissor",
    ["biased_rock_paper_scissor"] = "biased_rock_paper_scissor",
    ["biased_rock_paper_scissors"] = "biased_rock_paper_scissor",
    ["kuhn-poker"] = "kuhn_poker",
    ["kuhn_poker"] = "kuhn_poker",
}

local BUILTIN_DEFAULTS = {
    coin_flip = {
        observation_bits = 1,
        reward_bits = 1,
        agent_actions = 2,
        min_reward = 0,
        max_reward = 1,
    },
    ctw_test = {
        observation_bits = 1,
        reward_bits = 1,
        agent_actions = 2,
        min_reward = 0,
        max_reward = 1,
    },
    biased_rock_paper_scissor = {
        observation_bits = 2,
        reward_bits = 2,
        agent_actions = 3,
        min_reward = -1,
        max_reward = 1,
    },
    extended_tiger = {
        observation_bits = 3,
        reward_bits = 8,
        agent_actions = 4,
        min_reward = -100,
        max_reward = 30,
    },
    tic_tac_toe = {
        observation_bits = 18,
        reward_bits = 3,
        agent_actions = 9,
        min_reward = -3,
        max_reward = 2,
    },
    kuhn_poker = {
        observation_bits = 3,
        reward_bits = 3,
        agent_actions = 2,
        min_reward = -2,
        max_reward = 2,
    },
}

local function legacy_max_order(root)
    local n = as_int(root.rate_backend_max_order, "rate_backend_max_order")
        or as_int(root.max_order, "max_order")
        or as_int(root.rosa_max_order, "rosa_max_order")
        or 20
    if n < 1 then
        return 1
    end
    return n
end

local function mc_predictor_from_algorithm(root, interface)
    local algo = lower(root.algorithm or "ctw")
    local ct_depth = int_with_default_min(root.ct_depth, "ct_depth", 20, 1)

    if algo == "ctw" or algo == "fac-ctw" then
        local percept_bits = interface.observation_bits * math.max(interface.observation_stream_len, 1)
            + interface.reward_bits
        return {
            predictor = {
                kind = "fac-ctw",
                base_depth = ct_depth,
                num_percept_bits = percept_bits,
                encoding_bits = 8,
            },
        }
    end

    if algo == "ac-ctw" or algo == "ctw-context-tree" then
        return {
            predictor = { kind = "ctw", depth = ct_depth },
        }
    end

    if algo == "rosa" or algo == "rosaplus" then
        return {
            predictor = { kind = "rosaplus", max_order = as_int(root.rosa_max_order, "rosa_max_order") or legacy_max_order(root) or -1 },
        }
    end

    if algo == "rwkv" or algo == "rwkv7" then
        local method = root.rwkv_method
        if method ~= nil then
            return {
                predictor = { kind = "rwkv7", method = method },
            }
        end
        local model_path = root.rwkv_model_path
        if not model_path then
            die("legacy algorithm=rwkv requires rwkv_model_path/rwkv_method")
        end
        return {
            predictor = { kind = "rwkv7", model_path = model_path },
        }
    end

    if algo == "mamba" or algo == "mamba1" then
        local method = root.mamba_method
        if method ~= nil then
            return {
                predictor = { kind = "mamba", method = method },
            }
        end
        local model_path = root.mamba_model_path
        if not model_path then
            die("legacy algorithm=mamba requires mamba_model_path/mamba_method")
        end
        return {
            predictor = { kind = "mamba", model_path = model_path },
        }
    end

    if algo == "zpaq" then
        return {
            predictor = {
                kind = "zpaq",
                method = tostring(root.zpaq_method or "1"),
            },
        }
    end

    die("unknown legacy MC-AIXI algorithm '" .. tostring(root.algorithm) .. "'")
end

local function aiqi_predictor_from_algorithm(root, return_bits)
    local algo = lower(root.algorithm or "ac-ctw")
    local ct_depth = int_with_default_min(root.ct_depth, "ct_depth", 20, 1)

    if algo == "ctw" or algo == "ac-ctw" or algo == "ctw-context-tree" then
        return {
            predictor = { kind = "ctw", depth = ct_depth },
        }
    end

    if algo == "fac-ctw" then
        return {
            predictor = {
                kind = "fac-ctw",
                base_depth = ct_depth,
                num_percept_bits = return_bits,
                encoding_bits = 8,
            },
        }
    end

    if algo == "rosa" or algo == "rosaplus" then
        return {
            predictor = { kind = "rosaplus", max_order = as_int(root.rosa_max_order, "rosa_max_order") or legacy_max_order(root) or -1 },
        }
    end

    if algo == "rwkv" or algo == "rwkv7" then
        local method = root.rwkv_method
        if method ~= nil then
            return {
                predictor = { kind = "rwkv7", method = method },
            }
        end
        local model_path = root.rwkv_model_path
        if not model_path then
            die("legacy AIQI algorithm=rwkv requires rwkv_model_path/rwkv_method")
        end
        return {
            predictor = { kind = "rwkv7", model_path = model_path },
        }
    end

    if algo == "mamba" or algo == "mamba1" then
        local method = root.mamba_method
        if method ~= nil then
            return {
                predictor = { kind = "mamba", method = method },
            }
        end
        local model_path = root.mamba_model_path
        if not model_path then
            die("legacy AIQI algorithm=mamba requires mamba_model_path/mamba_method")
        end
        return {
            predictor = { kind = "mamba", model_path = model_path },
        }
    end

    if algo == "zpaq" then
        die("legacy AIQI algorithm=zpaq was unsupported and cannot be converted")
    end

    die("unknown legacy AIQI algorithm '" .. tostring(root.algorithm) .. "'")
end

local function render_number(n)
    if n ~= n or n == math.huge or n == -math.huge then
        die("cannot encode non-finite number")
    end
    if n == 0 then
        return "0"
    end
    if n % 1 == 0 then
        return string.format("%.0f", n)
    end
    local encoded = cjson.encode(n)
    if encoded == "-0" then
        return "0"
    end
    return encoded
end

local function is_array(tbl)
    if type(tbl) ~= "table" then
        return false
    end
    local count = 0
    local max_key = 0
    for k, _ in pairs(tbl) do
        if type(k) ~= "number" or k < 1 or k % 1 ~= 0 then
            return false
        end
        if k > max_key then
            max_key = k
        end
        count = count + 1
    end
    return max_key == count
end

local function sorted_keys(tbl)
    local keys = {}
    for k, _ in pairs(tbl) do
        keys[#keys + 1] = k
    end
    table.sort(keys, function(a, b)
        return tostring(a) < tostring(b)
    end)
    return keys
end

local function render_json(value, indent)
    if value == cjson.null then
        return "null"
    end

    local t = type(value)
    if t == "nil" then
        return "null"
    end
    if t == "boolean" then
        return value and "true" or "false"
    end
    if t == "number" then
        return render_number(value)
    end
    if t == "string" then
        return cjson.encode(value)
    end
    if t ~= "table" then
        die("unsupported JSON value type: " .. t)
    end

    if is_array(value) then
        if #value == 0 then
            return "[]"
        end
        local pieces = {"["}
        for i = 1, #value do
            pieces[#pieces + 1] = string.rep(" ", indent + 2)
                .. render_json(value[i], indent + 2)
                .. (i < #value and "," or "")
        end
        pieces[#pieces + 1] = string.rep(" ", indent) .. "]"
        return table.concat(pieces, "\n")
    end

    local keys = sorted_keys(value)
    if #keys == 0 then
        return "{}"
    end

    local pieces = {"{"}
    for i, key in ipairs(keys) do
        local rendered_key = cjson.encode(tostring(key))
        local rendered_value = render_json(value[key], indent + 2)
        pieces[#pieces + 1] = string.rep(" ", indent + 2)
            .. rendered_key
            .. ": "
            .. rendered_value
            .. (i < #keys and "," or "")
    end
    pieces[#pieces + 1] = string.rep(" ", indent) .. "}"
    return table.concat(pieces, "\n")
end

local function convert_legacy(root, input_path)
    local env_raw = lower(root.environment or "coin-flip")
    if env_raw == "external" then
        return nil, "External configs were deprecated"
    end

    local assets = make_asset_registry()
    local base_dir = dirname(input_path)

    local environment
    local env_defaults

    if env_raw == "vm" then
        local vm_env, vm_info = convert_legacy_vm_environment(root, base_dir, assets)
        environment = vm_env
        local min_reward, max_reward = signed_reward_bounds(vm_info.reward_bits)
        env_defaults = {
            observation_bits = vm_info.observation_bits,
            reward_bits = vm_info.reward_bits,
            agent_actions = vm_info.action_count,
            min_reward = min_reward,
            max_reward = max_reward,
        }
    else
        local builtin = BUILTIN_ENV[env_raw]
        if not builtin then
            die("unknown legacy environment '" .. tostring(root.environment) .. "'")
        end
        environment = {
            kind = "builtin",
            name = builtin,
        }
        env_defaults = BUILTIN_DEFAULTS[builtin]
        if env_defaults == nil then
            die("internal error: missing builtin defaults for '" .. builtin .. "'")
        end
    end

    local observation_stream_len = parse_observation_stream_len_for_env(root, env_raw)
    local observation_key_mode = parse_observation_key_mode_for_env(root, env_raw)

    local observation_bits = as_int(root.observation_bits, "observation_bits")
        or env_defaults.observation_bits
    if observation_bits < 1 then
        observation_bits = env_defaults.observation_bits
    end
    local reward_bits = as_int(root.reward_bits, "reward_bits")
        or env_defaults.reward_bits
    if reward_bits < 1 then
        reward_bits = env_defaults.reward_bits
    end

    local min_reward = env_defaults.min_reward
    local max_reward = env_defaults.max_reward
    local reward_offset = as_int(root.reward_offset, "reward_offset")
    if reward_offset == nil then
        reward_offset = math.max(0, -min_reward)
    end
    if reward_offset == 0 then
        reward_offset = 0
    end

    local interface = {
        observation_bits = observation_bits,
        observation_stream_len = observation_stream_len,
        observation_key_mode = observation_key_mode,
        reward_bits = reward_bits,
        agent_actions = int_with_default_min(root.agent_actions, "agent_actions", env_defaults.agent_actions, 1),
        min_reward = min_reward,
        max_reward = max_reward,
        reward_offset = reward_offset,
    }

    local planner_raw = lower(root.planner or root.solver or "mc-aixi")
    local planner_kind
    if planner_raw == "mc-aixi" or planner_raw == "mc_aixi" then
        planner_kind = "mc_aixi"
    elseif planner_raw == "aiqi" then
        planner_kind = "aiqi_discounted"
    else
        die("unsupported legacy planner/solver '" .. tostring(root.planner or root.solver) .. "'")
    end

    local run_seed = as_int(root.random_seed, "random_seed")
        or as_int(root.rng_seed, "rng_seed")

    local controller
    local random_seed = run_seed

    if planner_kind == "mc_aixi" then
        local predictor_override = nil
        if root.rate_backend ~= nil then
            predictor_override = backend_from_legacy_cfg(
                root.rate_backend,
                root,
                base_dir,
                observation_bits,
                reward_bits
            )
        end

        local predictor_pair
        if predictor_override ~= nil then
            if predictor_override.kind == "rosaplus" and predictor_override.max_order == nil then
                predictor_override.max_order = legacy_max_order(root) or -1
            end
            predictor_pair = {
                predictor = predictor_override,
            }
        else
            predictor_pair = mc_predictor_from_algorithm(root, interface)
        end

        controller = {
            kind = "mc_aixi",
            predictor = predictor_pair.predictor,
            agent_horizon = int_with_default_min(root.agent_horizon, "agent_horizon", 3, 1),
            num_simulations = int_with_default_min(root.num_simulations, "num_simulations", 50, 1),
            exploration_exploitation_ratio = positive_num_with_default(root.exploration_exploitation_ratio, "exploration_exploitation_ratio", 1.4),
            discount_gamma = closed_unit_num_with_default(root.discount_gamma, "discount_gamma", 1.0),
        }

        local planner_seed = as_int(root.mcaixi_random_seed, "mcaixi_random_seed")
        if planner_seed ~= nil and random_seed == nil then
            random_seed = planner_seed
        end
    else
        local predictor_override = nil
        if root.aiqi_rate_backend ~= nil then
            predictor_override = backend_from_legacy_cfg(
                root.aiqi_rate_backend,
                root,
                base_dir,
                observation_bits,
                reward_bits
            )
        elseif root.rate_backend ~= nil then
            predictor_override = backend_from_legacy_cfg(
                root.rate_backend,
                root,
                base_dir,
                observation_bits,
                reward_bits
            )
        end

        local discount_gamma
        if root.discount_gamma == nil then
            discount_gamma = 0.99
        else
            discount_gamma = as_num(root.discount_gamma, "discount_gamma")
        end

        local return_horizon = int_with_default_min(
            root.return_horizon ~= nil and root.return_horizon or root.agent_horizon,
            "return_horizon",
            3,
            1
        )
        local return_bins_raw = as_int(root.return_bins, "return_bins")
            or as_int(root.aiqi_bins, "aiqi_bins")
            or 16
        local return_bins = next_power_of_two(return_bins_raw)
        local augmentation_period = int_with_default_min(as_int(root.augmentation_period, "augmentation_period")
            or as_int(root.aiqi_period, "aiqi_period")
            or as_int(root.return_horizon, "return_horizon")
            or as_int(root.agent_horizon, "agent_horizon")
            or 3, "augmentation_period", 3, return_horizon)

        local predictor_pair
        if predictor_override ~= nil then
            if predictor_override.kind == "rosaplus" and predictor_override.max_order == nil then
                predictor_override.max_order = legacy_max_order(root) or -1
            end
            predictor_pair = {
                predictor = predictor_override,
            }
        else
            predictor_pair = aiqi_predictor_from_algorithm(root, return_bins)
        end

        controller = {
            kind = "aiqi_discounted",
            predictor = predictor_pair.predictor,
            discount_gamma = open_unit_num_with_default(discount_gamma, "discount_gamma", 0.99),
            return_horizon = return_horizon,
            return_bins = return_bins,
            augmentation_period = augmentation_period,
            history_prune_keep_steps = (function()
                local keep = as_int(root.history_prune_keep_steps, "history_prune_keep_steps")
                if keep == nil then
                    return nil
                end
                if keep < 0 then
                    return 0
                end
                return keep
            end)(),
            baseline_exploration = open_unit_num_with_default(as_num(root.baseline_exploration, "baseline_exploration")
                or as_num(root.tau, "tau")
                or 0.01, "baseline_exploration", 0.01),
        }

        local planner_seed = as_int(root.aiqi_random_seed, "aiqi_random_seed")
        if planner_seed ~= nil and random_seed == nil then
            random_seed = planner_seed
        end
    end

    local vm_perf_only = as_bool(root.vm_perf_only)
    local runtime = {
        random_seed = random_seed,
        log_every = int_with_default_min(root.log_every, "log_every", 1, 1),
        perf = as_bool(root.perf),
        vm_perf_only = vm_perf_only,
        explore_epsilon = nonnegative_num_with_default(root.explore_epsilon, "explore_epsilon", 0.0),
        explore_gamma = positive_num_with_default(root.explore_gamma, "explore_gamma", 1.0),
    }

    if vm_perf_only then
        local perf_cycles = as_int(root.perf_cycles, "perf_cycles")
        if perf_cycles ~= nil and perf_cycles < 1 then
            perf_cycles = 1
        end
        local terminate_lifetime = as_int(root["terminate-lifetime"], "terminate-lifetime")
        if terminate_lifetime ~= nil and terminate_lifetime < 1 then
            terminate_lifetime = 1
        end
        runtime.terminate_lifetime = perf_cycles or terminate_lifetime or 1000
    else
        local learn_cycles = as_int(root.learn_cycles, "learn_cycles")
        if learn_cycles ~= nil and learn_cycles < 0 then
            learn_cycles = 0
        end
        local eval_cycles = as_int(root.eval_cycles, "eval_cycles")
        if eval_cycles ~= nil and eval_cycles < 0 then
            eval_cycles = 0
        end
        runtime.learn_cycles = learn_cycles
        runtime.eval_cycles = eval_cycles
        runtime.terminate_lifetime = int_with_default_min(root["terminate-lifetime"], "terminate-lifetime", 20, 1)
    end

    local doc = {
        schema_version = 1,
        kind = "planner_run",
        assets = assets.list(),
        environment = environment,
        interface = interface,
        controller = controller,
        runtime = runtime,
    }

    return doc, nil
end

local input_path = arg[1]
if not input_path or input_path == "" then
    die("usage: " .. (arg[0] or "legacy_aixi_convert.lua") .. " <legacy-config.json|->")
end

local raw
if input_path == "-" then
    raw = read_stdin()
    input_path = "stdin.json"
else
    raw = read_file(input_path)
end

local ok, value = pcall(cjson.decode, raw)
if not ok then
    die("invalid JSON input: " .. tostring(value))
end
if type(value) ~= "table" then
    die("top-level JSON value must be an object")
end

if value.schema_version ~= nil and value.kind ~= nil then
    io.write(render_json(value, 0))
    io.write("\n")
    os.exit(0)
end

local converted, message = convert_legacy(value, input_path)
if converted == nil then
    io.write(message)
    os.exit(0)
end

io.write(render_json(converted, 0))
io.write("\n")

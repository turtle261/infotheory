#!/usr/bin/env luajit

local ok_cjson, cjson = pcall(require, "cjson")
if not ok_cjson then
    io.stderr:write("ERROR: lua-cjson is required (module 'cjson' not found)\n")
    os.exit(1)
end

local function die(msg)
    io.stderr:write("ERROR: " .. msg .. "\n")
    os.exit(1)
end

local function is_null(value)
    return value == nil or value == cjson.null
end

local function strip_ansi(s)
    return (s:gsub("\27%[[%d;?]*[ -/]*[@-~]", ""))
end

local function trim(s)
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function comma_int(value)
    if is_null(value) then
        return "-"
    end
    local s = string.format("%.0f", value)
    local sign = ""
    if s:sub(1, 1) == "-" then
        sign = "-"
        s = s:sub(2)
    end
    local rev = s:reverse():gsub("(%d%d%d)", "%1,")
    local out = rev:reverse():gsub("^,", "")
    return sign .. out
end

local function short_count(value)
    if is_null(value) then
        return "-"
    end
    local abs_value = math.abs(value)
    if abs_value >= 1e9 then
        return string.format("%.2fG", value / 1e9)
    end
    if abs_value >= 1e6 then
        return string.format("%.2fM", value / 1e6)
    end
    if abs_value >= 1e3 then
        return string.format("%.1fk", value / 1e3)
    end
    return string.format("%.0f", value)
end

local function human_bytes(bytes)
    if is_null(bytes) then
        return "-"
    end
    local units = {"B", "KiB", "MiB", "GiB", "TiB"}
    local value = bytes
    local unit = 1
    while math.abs(value) >= 1024 and unit < #units do
        value = value / 1024
        unit = unit + 1
    end
    if unit == 1 then
        return string.format("%.0f %s", value, units[unit])
    end
    if math.abs(value) >= 100 then
        return string.format("%.1f %s", value, units[unit])
    end
    return string.format("%.2f %s", value, units[unit])
end

local function human_seconds(seconds)
    if is_null(seconds) then
        return "-"
    end
    if seconds < 60 then
        return string.format("%.2fs", seconds)
    end
    local minutes = math.floor(seconds / 60)
    local rest = seconds - minutes * 60
    if minutes < 60 then
        return string.format("%dm%05.2fs", minutes, rest)
    end
    local hours = math.floor(minutes / 60)
    minutes = minutes - hours * 60
    return string.format("%dh%02dm%05.2fs", hours, minutes, rest)
end

local function percent(numer, denom)
    if is_null(numer) or is_null(denom) or denom == 0 then
        return "-"
    end
    return string.format("%.1f%%", 100.0 * numer / denom)
end

local function number_or_nil(value)
    if is_null(value) then
        return nil
    end
    return value
end

local function predicted_archive_bytes(snapshot)
    local bits = number_or_nil(snapshot.bits)
    if bits == nil then
        return nil
    end
    return bits / 8.0
end

local function snapshot_row(snapshot)
    local telemetry = snapshot.telemetry or {}
    local rss = snapshot.rss or {}
    local bpb = number_or_nil(snapshot.bits_per_byte)
    local archive = predicted_archive_bytes(snapshot)
    local hwm = number_or_nil(rss.vm_hwm_bytes) or number_or_nil(rss.vm_rss_bytes)
    return string.format(
        "%-10s %-10s %-10s %-11s %-11s %-11s %-10s %-10s %-10s %-9s %-9s %-7s",
        short_count(snapshot.bytes_seen),
        human_seconds(snapshot.elapsed_seconds),
        bpb and string.format("%.6f", bpb) or "-",
        archive and human_bytes(archive) or "-",
        human_bytes(hwm),
        human_bytes(telemetry.total_bytes),
        short_count(telemetry.nodes_len),
        short_count(telemetry.segments_len),
        short_count(telemetry.segment_bits),
        short_count(telemetry.history_segments),
        short_count(telemetry.history_invert_segments),
        telemetry.trees and tostring(#telemetry.trees) or "-"
    )
end

local function print_snapshot_header()
    print(string.format(
        "%-10s %-10s %-10s %-11s %-11s %-11s %-10s %-10s %-10s %-9s %-9s %-7s",
        "bytes", "elapsed", "bits/B", "ideal_out", "rss_hwm", "reserved", "nodes", "segments", "seg_bits", "histSeg", "invHist", "trees"
    ))
    print(string.rep("-", 132))
end

local function arena_payload_bytes(telemetry)
    local node_bytes = (telemetry.nodes_len or 0) * 32
    local segment_bytes = (telemetry.segments_len or 0) * 40
    return node_bytes + segment_bytes
end

local function explicit_node_equivalent_bytes(telemetry)
    local logical_nodes = (telemetry.nodes_len or 0) + (telemetry.segment_bits or 0)
    return logical_nodes * 32
end

local function print_kv(label, value)
    print(string.format("  %-30s %s", label .. ":", value))
end

local function print_summary(final, snapshots)
    local telemetry = final.telemetry or {}
    local rss = final.rss or {}
    local archive = predicted_archive_bytes(final)
    local logical_nodes = (telemetry.nodes_len or 0) + (telemetry.segment_bits or 0)
    local payload_bytes = arena_payload_bytes(telemetry)
    local explicit_bytes = explicit_node_equivalent_bytes(telemetry)
    local capacity_arena_bytes = (telemetry.nodes_capacity or 0) * 32 + (telemetry.segments_capacity or 0) * 40
    local capacity_slack = capacity_arena_bytes - payload_bytes
    local saved_vs_explicit = explicit_bytes - payload_bytes
    local history_payload_segments = (telemetry.history_segments or 0) + (telemetry.history_invert_segments or 0)

    print("")
    print("Final CTW profile summary")
    print(string.rep("=", 72))
    print_kv("snapshots", comma_int(snapshots))
    print_kv("mode", tostring(final.mode or "-"))
    print_kv("base depth", tostring(final.depth or telemetry.base_depth or "-"))
    print_kv("bytes seen", comma_int(final.bytes_seen))
    print_kv("elapsed", human_seconds(final.elapsed_seconds))
    if not is_null(final.bits_per_byte) then
        print_kv("rate", string.format("%.9f bits/byte", final.bits_per_byte))
    end
    if archive ~= nil then
        print_kv("ideal archive payload", human_bytes(archive) .. " (" .. comma_int(archive) .. " bytes)")
    end
    print_kv("RSS current", human_bytes(rss.vm_rss_bytes))
    print_kv("RSS high-water", human_bytes(rss.vm_hwm_bytes))
    print_kv("telemetry reserved total", human_bytes(telemetry.total_bytes))
    print_kv("tree arena reserved", human_bytes(telemetry.tree_bytes))
    print_kv("shared history reserved", human_bytes(telemetry.shared_history_bytes))
    print_kv("shared log cache reserved", human_bytes(telemetry.shared_log_cache_bytes))
    print_kv("history length", comma_int(telemetry.shared_history_len_bits) .. " bits")
    print_kv("history capacity", comma_int(telemetry.shared_history_capacity_bits) .. " bits")
    print_kv("nodes", comma_int(telemetry.nodes_len) .. " / cap " .. comma_int(telemetry.nodes_capacity) .. " (" .. percent(telemetry.nodes_len, telemetry.nodes_capacity) .. " full)")
    print_kv("segments", comma_int(telemetry.segments_len) .. " / cap " .. comma_int(telemetry.segments_capacity) .. " (" .. percent(telemetry.segments_len, telemetry.segments_capacity) .. " full)")
    print_kv("arena payload at len", human_bytes(payload_bytes))
    print_kv("arena capacity slack", human_bytes(capacity_slack))
    print_kv("represented logical nodes", comma_int(logical_nodes))
    print_kv("explicit-node equivalent", human_bytes(explicit_bytes))
    print_kv("payload saved by segments", human_bytes(saved_vs_explicit))
    print_kv("exact segments", comma_int(telemetry.exact_segments))
    print_kv("history-anchor segments", comma_int(history_payload_segments))
    print_kv("const segments", comma_int(telemetry.const_segments))
    print_kv("segment bits", comma_int(telemetry.segment_bits))

    if history_payload_segments == 0 then
        print_kv("history-anchor audit", "none observed")
    else
        print_kv("history-anchor audit", "present; bounded ring history is not automatically exact")
    end

    if telemetry.trees ~= nil and #telemetry.trees > 0 then
        print("")
        print("Final per-tree arena")
        print(string.rep("-", 116))
        print(string.format(
            "%-4s %-6s %-10s %-10s %-8s %-10s %-10s %-8s %-10s %-8s %-8s",
            "bit", "depth", "nodes", "node_cap", "nfull", "segments", "seg_cap", "sfull", "seg_bits", "maxseg", "histSeg"
        ))
        print(string.rep("-", 116))
        for _, tree in ipairs(telemetry.trees) do
            local tree_hist = (tree.history_segments or 0) + (tree.history_invert_segments or 0)
            print(string.format(
                "%-4s %-6s %-10s %-10s %-8s %-10s %-10s %-8s %-10s %-8s %-8s",
                tostring(tree.bit_index),
                tostring(tree.max_depth),
                short_count(tree.nodes_len),
                short_count(tree.nodes_capacity),
                percent(tree.nodes_len, tree.nodes_capacity),
                short_count(tree.segments_len),
                short_count(tree.segments_capacity),
                percent(tree.segments_len, tree.segments_capacity),
                short_count(tree.segment_bits),
                tostring(tree.max_segment_len or "-"),
                short_count(tree_hist)
            ))
        end
    end
end

local snapshots = 0
local final = nil
local printed_header = false

for raw_line in io.lines() do
    local line = trim(strip_ansi(raw_line))
    if line ~= "" then
        local ok, decoded = pcall(cjson.decode, line)
        if not ok then
            die("failed to decode JSONL line " .. tostring(snapshots + 1) .. ": " .. tostring(decoded))
        end
        if decoded.kind ~= "ctw_profile_snapshot" then
            die("line " .. tostring(snapshots + 1) .. " is not a ctw_profile_snapshot")
        end
        if not printed_header then
            print_snapshot_header()
            printed_header = true
        end
        snapshots = snapshots + 1
        final = decoded
        print(snapshot_row(decoded))
        io.stdout:flush()
    end
end

if final == nil then
    die("no ctw-profile JSONL snapshots were read from stdin")
end

print_summary(final, snapshots)

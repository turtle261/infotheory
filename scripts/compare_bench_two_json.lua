#!/usr/bin/env luajit

local function die(msg)
	io.stderr:write(msg .. "\n")
	os.exit(1)
end

local CORE_OPERATIONS  = { h = true, compress = true, decompress = true }
local CORE_SUBJECTS    = { ppmd = true, ctw = true, rosa = true, rwkv7 = true, neural_mixture = true }
local CORE_SIZES       = { ["1048576"] = true, ["4194304"] = true, ["10000000"] = true }

local REQUIRED_COLUMNS = {
	"operation",
	"subject",
	"size_bytes",
	"compression_backend",
}

local OPTIONAL_PROVENANCE_COLUMNS = {
	"suite_spec_path",
	"suite_spec_sha256",
	"build_mode",
	"build_features",
}

local LEGACY_UNKNOWN = "__legacy_unknown__"

local baseline_path, candidate_path

do
	local i = 1
	while i <= #arg do
		if arg[i] == "--baseline" then
			i = i + 1
			if not arg[i] then
				die("--baseline requires a path")
			end
			baseline_path = arg[i]
		elseif string.sub(arg[i], 1, 2) == "--" then
			die("unknown option: " .. arg[i])
		elseif not candidate_path then
			candidate_path = arg[i]
		else
			die("unexpected argument: " .. arg[i])
		end
		i = i + 1
	end

	if not baseline_path or not candidate_path then
		die("usage: " .. (arg[0] or "compare_bench_two_json.lua") .. " --baseline <BASELINE_TSV> <CANDIDATE_TSV>")
	end
end

local SEP = "\0"
local EPS = 1e-12

local function canonicalize_subject(subject)
	if subject == "rwkv" then
		return "rwkv7"
	end
	return subject
end

local function chomp_cr(s)
	return (s:gsub("\r$", ""))
end

local function split_tsv(line)
	line = chomp_cr(line)
	local fields = {}
	for f in (line .. "\t"):gmatch("([^\t]*)\t") do
		fields[#fields + 1] = f
	end
	return fields
end

local function make_header_index(headers)
	local idx = {}
	for i, h in ipairs(headers) do
		if h ~= "" then
			idx[h] = i
		end
	end
	return idx
end

local function validate_required_columns(header_index, path)
	for _, name in ipairs(REQUIRED_COLUMNS) do
		if not header_index[name] then
			die("missing required TSV column in " .. path .. ": " .. name)
		end
	end
end

local function make_key(operation, subject, size_bytes, compression_backend)
	return operation .. SEP .. subject .. SEP .. size_bytes .. SEP .. compression_backend
end

local function format_compare_key(row)
	return "operation=" .. (row._operation or "")
		.. ", subject=" .. (row._subject or "")
		.. ", size_bytes=" .. (row._size_bytes or "")
		.. ", compression_backend=" .. (row._compression_backend or "")
end

local function duplicate_row_message(path, headers, existing, duplicate)
	local parts = {
		"duplicate comparison row in " .. path,
		"key: " .. format_compare_key(duplicate),
		"first line: " .. tostring(existing._line),
		"duplicate line: " .. tostring(duplicate._line),
	}

	local diffs = {}
	for _, h in ipairs(headers) do
		if h ~= "" then
			local a = existing[h] or ""
			local b = duplicate[h] or ""
			if a ~= b then
				diffs[#diffs + 1] = h .. ": " .. a .. " != " .. b
			end
		end
	end

	if #diffs > 0 then
		parts[#parts + 1] = "differing columns: " .. table.concat(diffs, "; ")
	end

	parts[#parts + 1] =
		"comparison rows must be unique by operation, subject, size_bytes, and compression_backend"
	if (existing.cpu or "") ~= (duplicate.cpu or "") then
		parts[#parts + 1] =
			"hint: this summary appears to mix CPU affinities; rerun with INFOTHEORY_BENCH_FRESH=1, "
			.. "set one INFOTHEORY_BENCH_CPU, or compare a summary filtered to one CPU"
	end

	return table.concat(parts, "\n")
end

local function load_rows(path)
	local f = io.open(path, "r")
	if not f then
		die("cannot open: " .. path)
	end

	local first = f:read("*l")
	if not first then
		f:close()
		die("empty file: " .. path)
	end

	local headers = split_tsv(first)
	local header_index = make_header_index(headers)
	validate_required_columns(header_index, path)

	local rows = {}
	local line_number = 1

	for line in f:lines() do
		line_number = line_number + 1
		if line ~= "" then
			local vals = split_tsv(line)
			local row = {}

			for j, h in ipairs(headers) do
				row[h] = vals[j] or ""
			end
			for _, name in ipairs(OPTIONAL_PROVENANCE_COLUMNS) do
				if not header_index[name] then
					row[name] = LEGACY_UNKNOWN
				end
			end

			local operation = row.operation or ""
			local subject = canonicalize_subject(row.subject or "")
			row.subject = subject
			local size_bytes = row.size_bytes or ""
			local compression_backend = row.compression_backend or ""

			local key = make_key(operation, subject, size_bytes, compression_backend)

			row._operation = operation
			row._subject = subject
			row._size_bytes = size_bytes
			row._compression_backend = compression_backend
			row._line = line_number

			if rows[key] then
				die(duplicate_row_message(path, headers, rows[key], row))
			end

			rows[key] = row
		end
	end

	f:close()
	return rows
end

local function collect_single_value(rows, path, field)
	local seen = {}
	for _, row in pairs(rows) do
		local value = chomp_cr(row[field] or "")
		if value ~= "" then
			seen[value] = true
		end
	end

	local count, only = 0, nil
	for value in pairs(seen) do
		count = count + 1
		only = value
	end

	if count == 0 then
		die("missing required provenance value in " .. path .. ": " .. field)
	end
	if count > 1 then
		die("multiple distinct provenance values in " .. path .. ": " .. field)
	end
	return only
end

local function collect_provenance(rows, path)
	return {
		suite_spec_path = collect_single_value(rows, path, "suite_spec_path"),
		suite_spec_sha256 = collect_single_value(rows, path, "suite_spec_sha256"),
		build_mode = collect_single_value(rows, path, "build_mode"),
		build_features = collect_single_value(rows, path, "build_features"),
	}
end

local function num(row, field)
	local v = row[field]
	if not v or v:match("^%s*$") then
		return nil
	end
	return tonumber(v)
end

local function int(row, field)
	local v = num(row, field)
	return v and math.floor(v) or nil
end

local function key_fields_from_row_pair(base, cand)
	local row = base or cand
	return row._operation, row._subject, row._size_bytes, row._compression_backend
end

local function is_core_row(base, cand)
	local op, subj, size = key_fields_from_row_pair(base, cand)
	return CORE_OPERATIONS[op] and CORE_SUBJECTS[subj] and CORE_SIZES[size]
end

local function compare(base, cand)
	local reasons = {}

	if cand.verified_all ~= "1" then
		reasons[#reasons + 1] = "verified_all != 1"
	end

	local br, cr = num(base, "real_seconds_median"), num(cand, "real_seconds_median")
	if br and cr then
		local lim = math.max(br * 1.05, br + 0.02)
		if cr > lim + EPS then
			reasons[#reasons + 1] = ("real_seconds_median %.12g > %.12g"):format(cr, lim)
		end
	end

	local bm, cm = num(base, "rss_kib_median"), num(cand, "rss_kib_median")
	if bm and cm then
		local lim = math.max(bm * 1.03, bm + 4096.0)
		if cm > lim + EPS then
			reasons[#reasons + 1] = ("rss_kib_median %.12g > %.12g"):format(cm, lim)
		end
	end

	local ba, ca = int(base, "archive_bytes_median"), int(cand, "archive_bytes_median")
	if ba and ca and ca > ba + 1 then
		reasons[#reasons + 1] = ("archive_bytes_median %d > %d"):format(ca, ba + 1)
	end

	local be, ce = num(base, "entropy_bpb_median"), num(cand, "entropy_bpb_median")
	if be and ce and ce > be + 1e-9 then
		reasons[#reasons + 1] = ("entropy_bpb_median %.12g > %.12g"):format(ce, be + 1e-9)
	end

	return reasons
end

local baseline_rows  = load_rows(baseline_path)
local candidate_rows = load_rows(candidate_path)
local baseline_provenance = collect_provenance(baseline_rows, baseline_path)
local candidate_provenance = collect_provenance(candidate_rows, candidate_path)

if baseline_provenance.suite_spec_sha256 ~= candidate_provenance.suite_spec_sha256 then
	if baseline_provenance.suite_spec_sha256 ~= LEGACY_UNKNOWN
		and candidate_provenance.suite_spec_sha256 ~= LEGACY_UNKNOWN then
		die("suite spec digest mismatch: baseline "
			.. baseline_provenance.suite_spec_sha256
			.. " != candidate "
			.. candidate_provenance.suite_spec_sha256)
	end
end

local key_set, keys  = {}, {}
for k in pairs(baseline_rows) do
	if not key_set[k] then
		key_set[k] = true
		keys[#keys + 1] = k
	end
end
for k in pairs(candidate_rows) do
	if not key_set[k] then
		key_set[k] = true
		keys[#keys + 1] = k
	end
end

local OP_ORDER = { h = 0, compress = 1, decompress = 2 }

table.sort(keys, function(a, b)
	local ra = baseline_rows[a] or candidate_rows[a]
	local rb = baseline_rows[b] or candidate_rows[b]

	local ao, as, az, ab = ra._operation, ra._subject, ra._size_bytes, ra._compression_backend
	local bo, bs, bz, bb = rb._operation, rb._subject, rb._size_bytes, rb._compression_backend

	local ai = OP_ORDER[ao] or 99
	local bi = OP_ORDER[bo] or 99
	if ai ~= bi then return ai < bi end
	if as ~= bs then return as < bs end

	local an = tonumber(az) or 0
	local bn = tonumber(bz) or 0
	if an ~= bn then return an < bn end

	return ab < bb
end)

local full_warnings, core_failures = 0, 0

print("baseline\t" .. baseline_path)
print("candidate\t" .. candidate_path)
print("baseline_suite_spec_path\t" .. baseline_provenance.suite_spec_path)
print("baseline_suite_spec_sha256\t" .. baseline_provenance.suite_spec_sha256)
print("baseline_build_mode\t" .. baseline_provenance.build_mode)
print("baseline_build_features\t" .. baseline_provenance.build_features)
print("candidate_suite_spec_path\t" .. candidate_provenance.suite_spec_path)
print("candidate_suite_spec_sha256\t" .. candidate_provenance.suite_spec_sha256)
print("candidate_build_mode\t" .. candidate_provenance.build_mode)
print("candidate_build_features\t" .. candidate_provenance.build_features)
print("scope\tstatus\toperation\tsubject\tsize_bytes\tcompression_backend\treasons")

for _, key in ipairs(keys) do
	local base = baseline_rows[key]
	local cand = candidate_rows[key]

	local op, subj, size, backend = key_fields_from_row_pair(base, cand)
	local scope = is_core_row(base, cand) and "core" or "full"

	local reasons
	if not base then
		reasons = { "missing baseline row" }
	elseif not cand then
		reasons = { "missing candidate row" }
	else
		reasons = compare(base, cand)
	end

	local status
	if #reasons == 0 then
		status = "OK"
	elseif scope == "core" then
		status = "FAIL"
		core_failures = core_failures + 1
	else
		status = "WARN"
		full_warnings = full_warnings + 1
	end

	print(table.concat({
		scope,
		status,
		op,
		subj,
		size,
		backend,
		(#reasons > 0) and table.concat(reasons, "; ") or "-"
	}, "\t"))
end

print(("summary\tcore_failures=%d\tfull_warnings=%d\trows=%d")
	:format(core_failures, full_warnings, #keys))

os.exit(core_failures > 0 and 1 or 0)

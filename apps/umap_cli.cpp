// --- phase4 additions --------------------------------------------------------
// `umap` command-line executable.
//
//   umap fit            --input X --save-model model.umap [--output emb] [...]
//   umap transform      --model model.umap --input Xnew --output emb
//   umap fit-transform  --input X --output emb [--save-model model.umap] [...]
//
// I/O formats (by file extension):
//   .npy       raw NumPy arrays: magic \x93NUMPY, format 1.0/2.0 headers,
//              '<f4' / '<f8' dtypes, C order, 1-D or 2-D (written as v1.0
//              '<f4' C-order 2-D). Minimal reader/writer implemented here.
//   otherwise  CSV (delimiters: comma / semicolon / whitespace). An optional
//              header row is autodetected (any non-numeric field in the first
//              row). Output CSV has no header, float32 round-trip precision.
//
// Models use the versioned "UMAPCPP1" binary format (umap::UMAP::save/load;
// see src/model_io.cpp): params + raw data + embedding + graph CSR + kNN
// arrays + prepared search index + RandomState bookkeeping, so `umap
// transform` queries a fitted model without refitting.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "umap/umap.hpp"

namespace {

// ------------------------------------------------------------------ matrices

struct Matrix {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<float> data;  // row-major
};

bool has_suffix(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                      });
}

// ----------------------------------------------------------------------- npy

Matrix read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open '" + path + "'");
    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("'" + path + "' is not a .npy file");
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t header_len = 0;
    if (ver[0] == 1) {
        unsigned char b[2];
        f.read(reinterpret_cast<char*>(b), 2);
        header_len = static_cast<uint32_t>(b[0]) |
                     (static_cast<uint32_t>(b[1]) << 8);
    } else if (ver[0] == 2 || ver[0] == 3) {
        unsigned char b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        header_len = static_cast<uint32_t>(b[0]) |
                     (static_cast<uint32_t>(b[1]) << 8) |
                     (static_cast<uint32_t>(b[2]) << 16) |
                     (static_cast<uint32_t>(b[3]) << 24);
    } else {
        throw std::runtime_error("'" + path + "': unsupported .npy version " +
                                 std::to_string(ver[0]));
    }
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    if (!f) throw std::runtime_error("'" + path + "': truncated .npy header");

    // 'descr': dtype string in quotes after the descr key.
    const auto value_in_quotes = [&](const char* key) -> std::string {
        const std::size_t k = header.find(key);
        if (k == std::string::npos) return "";
        std::size_t q1 = header.find_first_of("'\"", k + std::strlen(key));
        if (q1 == std::string::npos) return "";
        std::size_t q2 = header.find(header[q1], q1 + 1);
        if (q2 == std::string::npos) return "";
        return header.substr(q1 + 1, q2 - q1 - 1);
    };
    const std::string descr = value_in_quotes("'descr'");
    bool is_f8;
    if (descr == "<f4" || descr == "=f4" || descr == "|f4") {
        is_f8 = false;
    } else if (descr == "<f8" || descr == "=f8" || descr == "|f8") {
        is_f8 = true;
    } else {
        throw std::runtime_error(
            "'" + path + "': unsupported .npy dtype '" + descr +
            "' (need float32 '<f4' or float64 '<f8', little-endian)");
    }
    const std::size_t fo = header.find("'fortran_order'");
    if (fo != std::string::npos &&
        header.find("True", fo) != std::string::npos &&
        header.find("True", fo) < header.find("}", fo))
        throw std::runtime_error("'" + path +
                                 "': fortran_order=True is not supported "
                                 "(save with C order)");
    // shape tuple: integers between the parentheses after 'shape'.
    const std::size_t sh = header.find("'shape'");
    if (sh == std::string::npos)
        throw std::runtime_error("'" + path + "': .npy header without shape");
    const std::size_t p1 = header.find('(', sh);
    const std::size_t p2 = header.find(')', p1);
    if (p1 == std::string::npos || p2 == std::string::npos)
        throw std::runtime_error("'" + path + "': malformed .npy shape");
    std::vector<int64_t> shape;
    {
        std::string inside = header.substr(p1 + 1, p2 - p1 - 1);
        std::stringstream ss(inside);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            bool any_digit = false;
            for (char c : tok)
                if (std::isdigit(static_cast<unsigned char>(c)))
                    any_digit = true;
            if (any_digit) shape.push_back(std::stoll(tok));
        }
    }
    Matrix m;
    if (shape.size() == 1) {
        m.rows = shape[0];
        m.cols = 1;
    } else if (shape.size() == 2) {
        m.rows = shape[0];
        m.cols = shape[1];
    } else {
        throw std::runtime_error("'" + path + "': need a 1-D or 2-D array (got " +
                                 std::to_string(shape.size()) + " dims)");
    }
    const std::size_t count =
        static_cast<std::size_t>(m.rows) * static_cast<std::size_t>(m.cols);
    m.data.resize(count);
    if (is_f8) {
        std::vector<double> tmp(count);
        f.read(reinterpret_cast<char*>(tmp.data()),
               static_cast<std::streamsize>(count * sizeof(double)));
        if (static_cast<std::size_t>(f.gcount()) != count * sizeof(double))
            throw std::runtime_error("'" + path + "': truncated .npy data");
        for (std::size_t i = 0; i < count; ++i)
            m.data[i] = static_cast<float>(tmp[i]);
    } else {
        f.read(reinterpret_cast<char*>(m.data.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
        if (static_cast<std::size_t>(f.gcount()) != count * sizeof(float))
            throw std::runtime_error("'" + path + "': truncated .npy data");
    }
    return m;
}

void write_npy(const std::string& path, const float* data, int64_t rows,
               int64_t cols) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         std::to_string(rows) + ", " + std::to_string(cols) +
                         "), }";
    // Pad so magic(6) + version(2) + hlen(2) + header is a multiple of 64,
    // ending in '\n' (numpy format 1.0).
    const std::size_t base = 6 + 2 + 2;
    std::size_t total = base + header.size() + 1;
    const std::size_t pad = (64 - (total % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');
    const uint16_t hlen = static_cast<uint16_t>(header.size());
    f.write("\x93NUMPY", 6);
    f.put(static_cast<char>(1));
    f.put(static_cast<char>(0));
    f.put(static_cast<char>(hlen & 0xFF));
    f.put(static_cast<char>((hlen >> 8) & 0xFF));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(static_cast<std::size_t>(rows) *
                                         cols * sizeof(float)));
    if (!f) throw std::runtime_error("write to '" + path + "' failed");
}

// ----------------------------------------------------------------------- csv

std::vector<std::string> split_fields(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',' || c == ';' || c == '\t' || c == ' ') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool parse_float(const std::string& s, float& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return false;
    out = static_cast<float>(v);
    return true;
}

Matrix read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open '" + path + "'");
    Matrix m;
    std::string line;
    bool first_content_line = true;
    int64_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        const std::vector<std::string> fields = split_fields(line);
        if (fields.empty()) continue;
        std::vector<float> row;
        row.reserve(fields.size());
        bool ok = true;
        for (const std::string& s : fields) {
            float v;
            if (!parse_float(s, v)) {
                ok = false;
                break;
            }
            row.push_back(v);
        }
        if (!ok) {
            if (first_content_line) {
                // Header row (autodetected): skip it.
                first_content_line = false;
                continue;
            }
            throw std::runtime_error("'" + path + "' line " +
                                     std::to_string(lineno) +
                                     ": non-numeric field");
        }
        first_content_line = false;
        if (m.cols == 0) {
            m.cols = static_cast<int64_t>(row.size());
        } else if (static_cast<int64_t>(row.size()) != m.cols) {
            throw std::runtime_error(
                "'" + path + "' line " + std::to_string(lineno) + ": expected " +
                std::to_string(m.cols) + " columns, got " +
                std::to_string(row.size()));
        }
        m.data.insert(m.data.end(), row.begin(), row.end());
        ++m.rows;
    }
    if (m.rows == 0)
        throw std::runtime_error("'" + path + "': no data rows");
    return m;
}

void write_csv(const std::string& path, const float* data, int64_t rows,
               int64_t cols) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
    char buf[64];
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            std::snprintf(buf, sizeof(buf), "%.9g",
                          static_cast<double>(data[i * cols + j]));
            f << buf;
            if (j + 1 < cols) f << ',';
        }
        f << '\n';
    }
    if (!f) throw std::runtime_error("write to '" + path + "' failed");
}

Matrix read_matrix(const std::string& path) {
    return has_suffix(path, ".npy") ? read_npy(path) : read_csv(path);
}

void write_matrix(const std::string& path, const float* data, int64_t rows,
                  int64_t cols) {
    if (has_suffix(path, ".npy"))
        write_npy(path, data, rows, cols);
    else
        write_csv(path, data, rows, cols);
}

// ---------------------------------------------------------------------- args

[[noreturn]] void usage(int code) {
    std::fprintf(
        code == 0 ? stdout : stderr, "%s",
        "umap — UMAP dimension reduction (C++ port of lmcinnes/umap)\n"
        "\n"
        "Usage:\n"
        "  umap fit           --input X --save-model M [--output EMB] [opts]\n"
        "  umap transform     --model M --input XNEW --output EMB\n"
        "  umap fit-transform --input X --output EMB [--save-model M] [opts]\n"
        "\n"
        "I/O: .npy (float32/float64, C-order 1-D/2-D) or CSV (header row\n"
        "autodetected; output CSV is headerless). Models: UMAPCPP1 binary.\n"
        "\n"
        "Options (defaults mirror umap-learn):\n"
        "  --input FILE, --output FILE, --model FILE, --save-model FILE\n"
        "  --y FILE                 supervised targets (1 column; -1=unknown)\n"
        "  --n-neighbors INT        [15]\n"
        "  --n-components INT       [2]\n"
        "  --metric NAME            [euclidean] (any named metric or\n"
        "                           'precomputed' with a square distance X)\n"
        "  --metric-kwds K=V[,..]   keys: p=FLOAT, z=FLOAT, v=F1:F2:...\n"
        "  --min-dist FLOAT         [0.1]\n"
        "  --spread FLOAT           [1.0]\n"
        "  --n-epochs INT           [500 (200 for n>10000); +200 densmap]\n"
        "  --learning-rate FLOAT    [1.0]\n"
        "  --init NAME|FILE         [spectral] spectral|random|pca|\n"
        "                           tswspectral, or a matrix file\n"
        "  --random-state INT, --seed INT   [unseeded]\n"
        "  --transform-seed INT     [42]\n"
        "  --negative-sample-rate INT   [5]\n"
        "  --repulsion-strength FLOAT   [1.0]\n"
        "  --set-op-mix-ratio FLOAT     [1.0]\n"
        "  --local-connectivity FLOAT   [1.0]\n"
        "  --a FLOAT --b FLOAT      [fit from spread/min-dist]\n"
        "  --densmap                [off]\n"
        "  --dens-lambda FLOAT      [2.0]\n"
        "  --dens-frac FLOAT        [0.3]\n"
        "  --dens-var-shift FLOAT   [0.1]\n"
        "  --output-dens            [off]\n"
        "  --output-metric NAME     [euclidean]\n"
        "  --target-metric NAME     [categorical]\n"
        "  --target-weight FLOAT    [0.5]\n"
        "  --target-n-neighbors INT [-1 = n-neighbors]\n"
        "  --disconnection-distance FLOAT  [per-metric default]\n"
        "  --unique                 deduplicate identical rows [off]\n"
        "  --force-approximation    force the NN-descent branch [off]\n"
        "  --device cpu|gpu         [cpu] (gpu = Apple Metal; quality parity,\n"
        "                           falls back to CPU when unsupported)\n"
        "  --verbose\n"
        "  --help\n");
    std::exit(code);
}

struct Args {
    std::string cmd;
    std::string input, output, model, save_model, y_file;
    umap::UMAPParams params;
    std::string init_file;  // when --init names a file
};

double parse_double_arg(const std::string& flag, const std::string& val) {
    try {
        std::size_t pos = 0;
        const double d = std::stod(val, &pos);
        if (pos != val.size()) throw std::invalid_argument(val);
        return d;
    } catch (...) {
        throw std::runtime_error(flag + ": invalid number '" + val + "'");
    }
}

long parse_int_arg(const std::string& flag, const std::string& val) {
    try {
        std::size_t pos = 0;
        const long v = std::stol(val, &pos);
        if (pos != val.size()) throw std::invalid_argument(val);
        return v;
    } catch (...) {
        throw std::runtime_error(flag + ": invalid integer '" + val + "'");
    }
}

void parse_metric_kwds(const std::string& spec, umap::MetricKwds& kwds) {
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const std::size_t eq = item.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("--metric-kwds: expected K=V, got '" +
                                     item + "'");
        const std::string key = item.substr(0, eq);
        const std::string val = item.substr(eq + 1);
        if (key == "p") {
            kwds.p = parse_double_arg("--metric-kwds p", val);
        } else if (key == "z") {
            kwds.z = parse_double_arg("--metric-kwds z", val);
        } else if (key == "v" || key == "w" || key == "sigma") {
            kwds.v.clear();
            std::stringstream vs(val);
            std::string num;
            while (std::getline(vs, num, ':'))
                kwds.v.push_back(parse_double_arg("--metric-kwds v", num));
        } else {
            throw std::runtime_error("--metric-kwds: unknown key '" + key +
                                     "' (supported: p, z, v)");
        }
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) usage(2);
    a.cmd = argv[1];
    if (a.cmd == "--help" || a.cmd == "-h" || a.cmd == "help") usage(0);
    if (a.cmd != "fit" && a.cmd != "transform" && a.cmd != "fit-transform") {
        std::fprintf(stderr, "umap: unknown subcommand '%s'\n\n",
                     a.cmd.c_str());
        usage(2);
    }
    const auto need_value = [&](int i) -> std::string {
        if (i + 1 >= argc)
            throw std::runtime_error(std::string(argv[i]) +
                                     ": missing value");
        return argv[i + 1];
    };
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto next = [&]() { return need_value(i++); };
        if (flag == "--help" || flag == "-h") usage(0);
        else if (flag == "--input") a.input = next();
        else if (flag == "--output") a.output = next();
        else if (flag == "--model") a.model = next();
        else if (flag == "--save-model") a.save_model = next();
        else if (flag == "--y") a.y_file = next();
        else if (flag == "--n-neighbors")
            a.params.n_neighbors = static_cast<int>(parse_int_arg(flag, next()));
        else if (flag == "--n-components")
            a.params.n_components = static_cast<int>(parse_int_arg(flag, next()));
        else if (flag == "--metric") a.params.metric = next();
        else if (flag == "--metric-kwds")
            parse_metric_kwds(next(), a.params.metric_kwds);
        else if (flag == "--min-dist")
            a.params.min_dist = parse_double_arg(flag, next());
        else if (flag == "--spread")
            a.params.spread = parse_double_arg(flag, next());
        else if (flag == "--n-epochs")
            a.params.n_epochs = static_cast<int>(parse_int_arg(flag, next()));
        else if (flag == "--learning-rate")
            a.params.learning_rate = parse_double_arg(flag, next());
        else if (flag == "--init") {
            const std::string v = next();
            if (v == "spectral" || v == "random" || v == "pca" ||
                v == "tswspectral") {
                a.params.init = v;
            } else {
                a.init_file = v;  // ndarray init from a matrix file
            }
        } else if (flag == "--random-state" || flag == "--seed")
            a.params.random_state =
                static_cast<uint32_t>(parse_int_arg(flag, next()));
        else if (flag == "--transform-seed")
            a.params.transform_seed =
                static_cast<uint32_t>(parse_int_arg(flag, next()));
        else if (flag == "--negative-sample-rate")
            a.params.negative_sample_rate =
                static_cast<int>(parse_int_arg(flag, next()));
        else if (flag == "--repulsion-strength")
            a.params.repulsion_strength = parse_double_arg(flag, next());
        else if (flag == "--set-op-mix-ratio")
            a.params.set_op_mix_ratio = parse_double_arg(flag, next());
        else if (flag == "--local-connectivity")
            a.params.local_connectivity = parse_double_arg(flag, next());
        else if (flag == "--a")
            a.params.a = parse_double_arg(flag, next());
        else if (flag == "--b")
            a.params.b = parse_double_arg(flag, next());
        else if (flag == "--densmap") a.params.densmap = true;
        else if (flag == "--dens-lambda")
            a.params.dens_lambda = parse_double_arg(flag, next());
        else if (flag == "--dens-frac")
            a.params.dens_frac = parse_double_arg(flag, next());
        else if (flag == "--dens-var-shift")
            a.params.dens_var_shift = parse_double_arg(flag, next());
        else if (flag == "--output-dens") a.params.output_dens = true;
        else if (flag == "--output-metric") a.params.output_metric = next();
        else if (flag == "--target-metric") a.params.target_metric = next();
        else if (flag == "--target-weight")
            a.params.target_weight = parse_double_arg(flag, next());
        else if (flag == "--target-n-neighbors")
            a.params.target_n_neighbors =
                static_cast<int>(parse_int_arg(flag, next()));
        else if (flag == "--disconnection-distance")
            a.params.disconnection_distance = parse_double_arg(flag, next());
        else if (flag == "--unique") a.params.unique = true;
        else if (flag == "--force-approximation")
            a.params.force_approximation_algorithm = true;
        else if (flag == "--device") {
            const std::string v = next();
            if (v == "cpu") a.params.device = umap::Device::CPU;
            else if (v == "gpu") a.params.device = umap::Device::GPU;
            else
                throw std::runtime_error("--device: expected cpu|gpu, got '" +
                                         v + "'");
        } else if (flag == "--verbose") a.params.verbose = true;
        else {
            std::fprintf(stderr, "umap: unknown option '%s'\n\n",
                         flag.c_str());
            usage(2);
        }
    }
    return a;
}

// ----------------------------------------------------------------- commands

int run_fit(const Args& args, bool also_transform) {
    if (args.input.empty())
        throw std::runtime_error("--input is required");
    if (also_transform && args.output.empty())
        throw std::runtime_error("fit-transform requires --output");
    if (!also_transform && args.save_model.empty() && args.output.empty())
        throw std::runtime_error("fit requires --save-model (or --output)");

    const Matrix X = read_matrix(args.input);
    umap::UMAPParams params = args.params;

    std::optional<Matrix> init_mat;
    if (!args.init_file.empty()) {
        init_mat = read_matrix(args.init_file);
        if (init_mat->rows != X.rows)
            throw std::runtime_error(
                "--init file must have one row per input sample");
        params.init_array = init_mat->data;
        params.init_array_cols = init_mat->cols;
    }

    std::optional<Matrix> y_mat;
    const float* y = nullptr;
    if (!args.y_file.empty()) {
        y_mat = read_matrix(args.y_file);
        if (y_mat->cols != 1)
            throw std::runtime_error("--y must be a single column");
        if (y_mat->rows != X.rows)
            throw std::runtime_error("--y must have one row per input sample");
        y = y_mat->data.data();
    }

    umap::UMAP model(params);
    model.fit(X.data.data(), X.rows, X.cols, y);

    if (!args.output.empty()) {
        write_matrix(args.output, model.embedding().data(), X.rows,
                     params.n_components);
        if (params.verbose)
            std::fprintf(stderr, "umap: wrote %" PRId64 " x %d embedding to %s\n",
                         X.rows, params.n_components, args.output.c_str());
    }
    if (!args.save_model.empty()) {
        model.save(args.save_model);
        if (params.verbose)
            std::fprintf(stderr, "umap: saved model to %s\n",
                         args.save_model.c_str());
    }
    return 0;
}

int run_transform(const Args& args) {
    if (args.model.empty())
        throw std::runtime_error("transform requires --model");
    if (args.input.empty())
        throw std::runtime_error("--input is required");
    if (args.output.empty())
        throw std::runtime_error("transform requires --output");

    umap::UMAP model = umap::UMAP::load(args.model);
    const Matrix X = read_matrix(args.input);
    if (model.params().metric != "precomputed" && X.cols != model.n_features())
        throw std::runtime_error(
            "input has " + std::to_string(X.cols) + " columns; the model was "
            "fit on " + std::to_string(model.n_features()) + " features");
    const std::vector<float> emb = model.transform(X.data.data(), X.rows);
    write_matrix(args.output, emb.data(), X.rows,
                 model.params().n_components);
    if (model.params().verbose || args.params.verbose)
        std::fprintf(stderr, "umap: wrote %" PRId64 " x %d embedding to %s\n",
                     X.rows, model.params().n_components,
                     args.output.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.cmd == "fit") return run_fit(args, /*also_transform=*/false);
        if (args.cmd == "fit-transform")
            return run_fit(args, /*also_transform=*/true);
        return run_transform(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "umap: %s\n", e.what());
        return 1;
    }
}

// yolov8
#include <sched.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "cix_noe_standard_api.h"

// exit_on_noe_error: handle errors on noe function calls
inline void exit_on_noe_error(context_handler_t* ctx,
                               noe_status_t status,
                               const char* call_name) {

    if (status != NOE_STATUS_SUCCESS) {
        const char* msg = nullptr;
        noe_get_error_message(ctx, status, &msg);

        std::cerr << call_name << " failed (code "
                  << status << "): "
                  << (msg ? msg : "Unknown error")
                  << std::endl;

        std::exit(1);
    }
}

// noe_handle: RAII for context, graph, job
typedef uint64_t u64;
struct noe_handle {
    context_handler_t* ctx      = nullptr;
    u64                graph_id = 0;
    u64                job_id   = 0;

    ~noe_handle() {
        if (job_id)    noe_clean_job(ctx, job_id);
        if (graph_id)  noe_unload_graph(ctx, graph_id);
        if (ctx)       noe_deinit_context(ctx);
    }

    void init(const std::string &model_path) {
        exit_on_noe_error(ctx,
            noe_init_context(&ctx),
            "noe_init_context"
        );

        std::cout << "NOE context initialized" << std::endl;

        exit_on_noe_error(ctx,
            noe_load_graph(ctx, model_path.c_str(), &graph_id),
            "noe_load_graph"
        );

        std::cout << "Model/Graph loaded" << std::endl;
    }

    void create_job() {
        job_config_npu_t npu_cfg{};
        job_config_t     cfg{ &npu_cfg };

        exit_on_noe_error(ctx,
            noe_create_job(ctx, graph_id, &job_id, &cfg),
            "noe_create_job"
        );

        std::cout << "Created Job: " << job_id << std::endl;
    }
};


// pin_to_cores: sets cpu affinity of program to run on specified cores
void pin_to_cores(const std::vector<int>& cpus) {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    for (int cpu : cpus) {
        CPU_SET(cpu, &mask);
    }

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        perror("sched_setaffinity");
        std::cerr << "Warning: could not set CPU affinity" << std::endl;
    }
}


// data_type_to_string: convert NOE data type to string
static const char* data_type_to_string(noe_data_type_t dt) {
    switch(dt) {
        case NOE_DATA_TYPE_NONE:  return "NONE";
        case NOE_DATA_TYPE_BOOL:  return "BOOL";
        case NOE_DATA_TYPE_U8:    return "U8";
        case NOE_DATA_TYPE_S8:    return "S8";
        case NOE_DATA_TYPE_U16:   return "U16";
        case NOE_DATA_TYPE_S16:   return "S16";
        case NOE_DATA_TYPE_U32:   return "U32";
        case NOE_DATA_TYPE_S32:   return "S32";
        case NOE_DATA_TYPE_U64:   return "U64";
        case NOE_DATA_TYPE_S64:   return "S64";
        case NOE_DATA_TYPE_F16:   return "F16";
        case NOE_DATA_TYPE_F32:   return "F32";
        case NOE_DATA_TYPE_F64:   return "F64";
        case NOE_DATA_TYPE_BF16:  return "BF16";
        default:                  return "UNKNOWN";
    }
}

// preprocess_image: read image from filename, resize to tensor input
// size, then return as uint8 blob
cv::Mat preprocess_image(const std::string &path, int size) {

    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "Cannot read image: " << path << std::endl;
        std::exit(1);
    }

    cv::resize(img, img, {size, size});
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob,
                           1.0f,
                           {size, size},
                           {}, false, false, CV_8U);

    return blob;
}

// read_output: fetch and dequantize tensor
// the cixbuild compiler 24Q4 version outputs S16 whilst the
// 25Q1 version outputs F16, so we handle both types
std::vector<float> read_output(context_handler_t* ctx,
                               u64 job_id,
                               const tensor_desc_t &desc) {
    size_t bytes = desc.size;
    std::vector<float> out;

    if (desc.data_type == NOE_DATA_TYPE_S16) {
        size_t count = bytes / sizeof(int16_t);
        std::vector<int16_t> buf(count);

        exit_on_noe_error(ctx,
            noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 0, buf.data()),
            "noe_get_tensor"
        );

        out.reserve(count);
        for (auto v : buf) out.push_back(v / desc.scale);
    }
    else if (desc.data_type == NOE_DATA_TYPE_F16) {
        size_t count = bytes / sizeof(uint16_t);
        std::vector<uint16_t> buf(count);

        exit_on_noe_error(ctx,
            noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 0, buf.data()),
            "noe_get_tensor"
        );

        out.reserve(count);
        for (auto raw : buf) {
            __fp16 h16 = *reinterpret_cast<__fp16*>(&raw);
            out.push_back(static_cast<float>(h16));
        }
    }
    else {
        std::cerr << "Unsupported data type: " << desc.data_type << std::endl;
        std::exit(1);
    }
    return out;
}

// detection: represents a single final object detection
//   cls  : detected class ID
//   conf : confidence score for the detection
//   x1,y1: coordinates of the top-left corner of the bounding box
//   x2,y2: coordinates of the bottom-right corner of the bounding box
struct detection {
    int cls;
    float conf, x1, y1, x2, y2;
};

// postprocess: decode raw predictions and apply Non-Maximum Suppression (NMS)
std::vector<detection> postprocess(const std::vector<float> &pred,
                                   int rows, int cols,
                                   float conf_thr, float iou_thr) {

    // Create a temporary box struct to hold decoded coordinates and score
    struct box { float x1, y1, x2, y2, conf; int cls; };
    std::vector<box> candidates;
    candidates.reserve(rows);

    // Decode each row of raw predictions into boxes + confidence
    for (int i = 0; i < rows; ++i) {
        const float* base = pred.data() + i*cols;
        // Find the class with maximum raw score
        float best = -1e9f;
        int bi = -1;

        for (int j = 4; j < cols; ++j) {
            if (base[j] > best) {
                best = base[j];
                bi = j;
            }
        }
        // Skip low-confidence detections
        if (best <= conf_thr) continue;

        // Decode box center-size to corner coordinates
        float cx = base[0];
        float cy = base[1];
        float w  = base[2];
        float h  = base[3];
        float x1 = cx - w/2.0f;
        float y1 = cy - h/2.0f;
        float x2 = cx + w/2.0f;
        float y2 = cy + h/2.0f;

        // Save candidate: corners, confidence, and class index
        candidates.push_back({x1, y1, x2, y2, best, bi - 4});
    }

    // Group detections by class for per-class NMS
    std::map<int, std::vector<box>> by_class;
    for (auto &b : candidates) by_class[b.cls].push_back(b);

    std::vector<detection> dets;
    // Apply greedy NMS within each class
    for (auto &kv : by_class) {
        auto &boxes = kv.second;
        // Sort by descending confidence
        std::sort(boxes.begin(), boxes.end(),
                  [](auto &a, auto &b){ return a.conf > b.conf; });
        std::vector<bool> removed(boxes.size());

        for (size_t i = 0; i < boxes.size(); ++i) {
            if (removed[i]) continue;
            auto &b_i = boxes[i];
            // Keep this box
            dets.push_back({kv.first, b_i.conf, b_i.x1, b_i.y1, b_i.x2, b_i.y2});

            // Suppress overlapping boxes
            for (size_t j = i+1; j < boxes.size(); ++j) {
                if (removed[j]) continue;
                auto &b_j = boxes[j];
                // Compute intersection rectangle
                float xx1 = std::max(b_i.x1, b_j.x1);
                float yy1 = std::max(b_i.y1, b_j.y1);
                float xx2 = std::min(b_i.x2, b_j.x2);
                float yy2 = std::min(b_i.y2, b_j.y2);
                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float inter = w * h;
                // Union area = area1 + area2 - inter
                float uni = (b_i.x2 - b_i.x1) * (b_i.y2 - b_i.y1)
                          + (b_j.x2 - b_j.x1) * (b_j.y2 - b_j.y1)
                          - inter;
                // If IoU exceeds threshold, remove
                if (inter / uni > iou_thr) {
                    removed[j] = true;
                }
            }
        }
    }
    return dets;
}

// List of 80 COCO class names
static const std::vector<std::string> coco_names = {
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
    "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

int main(int argc, char **argv) {

    // pin to fast CPU cores only
    // {0,5,6,7,8,9,10,11} = all A720 cores at 2.2-2.5Ghz
    // {1,2,3,4} = A520 cores @ 1.8Ghz
    // {0, 11, 9, 10} = fastest A720 cores at >= 2.4Ghz
    pin_to_cores({0, 11, 9, 10});

    // check argument count
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " model.cix image.jpg boxThreshold nmsThreshold\n";
        std::exit(1);
    }

    // Initialize NOE NPU driver context and load graph/model
    noe_handle h;
    h.init(argv[1]);
    h.create_job();

    // query tensor counts and descriptors
    uint32_t in_cnt = 0, out_cnt = 0;
    exit_on_noe_error(h.ctx,
        noe_get_tensor_count(h.ctx, h.graph_id, NOE_TENSOR_TYPE_INPUT, &in_cnt),
        "noe_get_tensor_count(input)"
    );

    exit_on_noe_error(h.ctx,
        noe_get_tensor_count(h.ctx, h.graph_id, NOE_TENSOR_TYPE_OUTPUT, &out_cnt),
        "noe_get_tensor_count(output)"
    );

    std::cout << "Tensor Counts, Input=" << in_cnt << ", Output=" << out_cnt << std::endl;

    // print tensor descriptors
    tensor_desc_t in_desc;
    exit_on_noe_error(h.ctx,
        noe_get_tensor_descriptor(h.ctx, h.graph_id, NOE_TENSOR_TYPE_INPUT, 0, &in_desc),
        "noe_get_tensor_descriptor(input)"
    );

    std::cout << "Input tensor descriptor:\n"
              << "  id:          " << in_desc.id << "\n"
              << "  size:        " << in_desc.size << "\n"
              << "  scale:       " << in_desc.scale << "\n"
              << "  zero_point:  " << in_desc.zero_point << "\n"
              << "  data_type:   " << data_type_to_string(in_desc.data_type) << "\n";

    tensor_desc_t out_desc0;
    exit_on_noe_error(h.ctx,
        noe_get_tensor_descriptor(h.ctx, h.graph_id, NOE_TENSOR_TYPE_OUTPUT, 0, &out_desc0),
        "noe_get_tensor_descriptor(output)"
    );

    std::cout << "Output tensor descriptor:\n"
              << "  id:          " << out_desc0.id << "\n"
              << "  size:        " << out_desc0.size << "\n"
              << "  scale:       " << out_desc0.scale << "\n"
              << "  zero_point:  " << out_desc0.zero_point << "\n"
              << "  data_type:   " << data_type_to_string(out_desc0.data_type) << "\n";

    // Preprocess input image into blob
    cv::Mat blob = preprocess_image(argv[2], 640);

    // Load tensor data to NPU and time it
    auto t0 = std::chrono::high_resolution_clock::now();

    exit_on_noe_error(h.ctx,
        noe_load_tensor(h.ctx, h.job_id, 0, blob.data),
        "noe_load_tensor"
    );
    std::cout << "Tensor load time: " << std::chrono::duration<float, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0
               ).count() << " ms\n";

    // Run inference and time it
    t0 = std::chrono::high_resolution_clock::now();

    exit_on_noe_error(h.ctx,
        noe_job_infer_sync(h.ctx, h.job_id, 5000),
        "noe_job_infer_sync"
    );

    std::cout << "Inference sync time: " << std::chrono::duration<float, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0
               ).count() << " ms\n";

    // Fetch output tensor
    t0 = std::chrono::high_resolution_clock::now();
    auto pred = read_output(h.ctx, h.job_id, out_desc0);

    std::cout << "Fetch outputs time: " << std::chrono::duration<float, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0
               ).count() << " ms\n";

    //Reorder CHW -> row-major for postprocess
    const int ROWS = 8400, COLS = 84;
    std::vector<float> rm(ROWS * COLS);

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            rm[i * COLS + j] = pred[j * ROWS + i];
        }
    }

    // Post-process (NMS) and print detections
    auto dets = postprocess(
        rm, ROWS, COLS,
        std::stof(argv[3]),
        std::stof(argv[4])
    );

    for (auto &d : dets) {
        // Lookup object name from COCO labels and print
        std::string name;
        if (d.cls >= 0 && d.cls < (int)coco_names.size()) {
            name = coco_names[d.cls];
        } else {
            name = "unknown";
        }

        // output object detection
        std::cout << name << " "
            << std::fixed << std::setprecision(3) << d.conf
            << std::fixed << std::setprecision(0)
            << " ("
            << d.x1 << ","
            << d.y1 << ","
            << d.x2 << ","
            << d.y2
            << ")\n";
    }

    return 0;
}

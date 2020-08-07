#include<iostream>
#include<vector>
#include "tkdnn.h"
#include "test.h"
#include "DarknetParser.h"

// Convert Darknet Weights to TensorRT runtime instance
// You must provide a string path to the yolov4 cfg file AND a names file.
// The yolov4.cfg file must be the exact same one for training as it is used for inference
// AKA inferring at a lower size is NOT supported

int main(int argc, char* argv[]) {
    if(argc < 3){
        FatalError("Not enough parameters, must provide path to yolov4 layers, cfg, names");
    }
    std::string rt_file_name = "yolov4";
    std::string bin_path  = std::string(argv[1]);
    std::vector<std::string> input_bins = {
        bin_path + "/layers/input.bin"
    };
    std::vector<std::string> output_bins = {
        bin_path + "/debug/layer139_out.bin",
        bin_path + "/debug/layer150_out.bin",
        bin_path + "/debug/layer161_out.bin"
    };
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path  = std::string(argv[2]);
    std::string name_path = std::string(argv[3]);
    // downloadWeightsifDoNotExist(input_bins[0], bin_path, "https://cloud.hipert.unimore.it/s/d97CFzYqCPCp5Hg/download");

    // parse darknet network
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
    net->print();

    //convert network to tensorRT
    tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, net->getNetworkRTName(rt_file_name.c_str()));

    int ret = testInference(input_bins, output_bins, net, netRT);
    net->releaseLayers();
    delete net;
    delete netRT;
    return ret;
}

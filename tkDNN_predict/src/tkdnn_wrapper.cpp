#ifndef TKDNN_WRAPPER
#define TKDNN_WRAPPER

#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <mutex>

// opencv stuff should probably stay for image loading for debugging only
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkDNN/Yolo3Detection.h"

extern "C++" {
    #include "tkdnn.h" // tkdnn core libraries
    #include "tkDNN/DarknetParser.h" // darknet parser utils
}

// When compiling this wrapper, if your OPENCV2 is compiled with the contrib from opencv4, comment this in for GPU accelerated resize
// #define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

// This code was adapted from DetectionNN.h
// Note 1: The word "frame" and "image" are used iterchangably in this file. Both refer to an image file.

// For testing purposes, we are trying out Apex, with 38 classes, bs=1

std::vector<cv::Mat> loadImagesFromList(std::string listPath) {
    cv::Mat image;
    std::vector<cv::Mat> loaded_images;

    std::ifstream input_stream;
    input_stream.open(listPath);
    std::string line;

    while( getline(input_stream, line) ){
        image = cv::imread(line, cv::IMREAD_COLOR);   // Read the file
        if(! image.data ) {                             // Check for invalid input
            FatalError("Image not valid")
        }
        loaded_images.push_back(image);
    }
    return loaded_images;
}

// Write performance to file
void dumpPerformanceStats(tk::dnn::DetectionNN* detNN) {
    std::ofstream outputfile;
    std::cout << "Writing " << detNN->stats.size() << " performance numbers to performance.csv\n";
    outputfile.open("performance.csv");
    for(int i=0; i<detNN->stats.size(); i++){
        outputfile << detNN->stats[i] << ",\n";
    }
    outputfile.close();
}

int main(int argc, char* argv[]){
    if(argc < 2){
        FatalError("Not enough parameters, must provide path to yolov4 rt file and text file with list of images");
    }
    int batch_size = 4;
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::DetectionNN* network = &yolo;
    std::vector<cv::Mat> dnnInput;
    std::vector<cv::Mat> inference_batch;

    network->init(argv[1], 38, batch_size);
    std::cout << "Yolo initialized" << std::endl;
    
    dnnInput = loadImagesFromList(argv[2]);  // Add the image to a vector of frames 

    std::cout << "Loaded " << dnnInput.size() << " images from file " << argv[2] << std::endl;

    // Pop the first 4 items off the loaded images up to a batch_size, then feed batch into NN
    int num_batches = (int)(dnnInput.size()/batch_size); // Calculate the number of batches, rounded down in case things go awry
    for (int i = 0; i < num_batches; i++ ) {
        inference_batch.clear(); // clear out the last batch
        for (int b = 4*i; b < batch_size*(i+1); b++) {
            inference_batch.push_back(dnnInput[b].clone());
        }

        std::cout << "Inferring on batch #" << i << " containing " << inference_batch.size() << " images\n";

        network->update(inference_batch, batch_size); // do the inference 

        std::vector<std::vector<tk::dnn::box>> boxes = network->getRawBoundingBoxes();

        std::cout << "Number of BBs: " << boxes[0].size() << std::endl;
        
        for (int i = 0; i < boxes[0].size(); i++) {
            std::cout << "Box " << i << " contents: \n";
            boxes[0][i].print();
        }
    }
    // std::cout << "Inferred in " << *network->stats.begin() << " milliseconds\n";
    dumpPerformanceStats(network);
}

#endif /* TKDNN_WRAPPER*/

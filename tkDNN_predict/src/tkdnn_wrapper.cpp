#ifndef TKDNN_WRAPPER
#define TKDNN_WRAPPER

#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <mutex>
#include <math.h>       /* ceil */

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
    if(argc < 4){
        FatalError("Not enough parameters, must provide in order: path to yolov4 rt file, text file with list of images, batch size, number of classes");
    }
    // Set batch size from command lines
    float batch_size = atoi(argv[3]);

    // Initialize Yolo detection objects
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::DetectionNN* network = &yolo;
    std::vector<cv::Mat> dnnInput;
    std::vector<cv::Mat> inference_batch;

    // Initialize the network. Init takes in three parameters: string path to .rt file, number of classes, batch_size
    network->init(argv[1], atoi(argv[4]), batch_size);
    std::cout << "Yolo initialized" << std::endl;
    
    // Load in images from a text file. Regardless of what happens, your desired images you want to infer on needs to
    // get into a vector of cv::Mat 
    dnnInput = loadImagesFromList(argv[2]);  // Add the image to a vector of frames 

    std::cout << "Loaded " << dnnInput.size() << " images from file " << argv[2] << std::endl;

    // Pop the first 4 items off the loaded images up to a batch_size, then feed batch into NN
    float dnnInputSize = dnnInput.size();
    int num_batches = std::ceil(dnnInputSize/batch_size); // Calculate the number of batches, rounded up
    std::cout << "Number of batches: " << num_batches << std::endl;
    for (int i = 0; i < num_batches; i++ ) {
        // inference_batch is a vector of size batch_size containing cv::mats that we will infer on
        inference_batch.clear(); // clear out the last batch
        for (int b = batch_size*i; b < batch_size*(i+1); b++) {
            if(b < dnnInput.size()) { // Do not add if we cannot make a full batch
                inference_batch.push_back(dnnInput[b].clone());
            }
        }

        std::cout << "Inferring on batch #" << i << " containing " << inference_batch.size() << " images\n";

        // do the inference on inference_batch. 
        network->update(inference_batch, inference_batch.size()); 

        // The raw bounding boxes are saved as a vector of vectors. 
        // The outer vector is a vector of bounding boxes of each batch, indexed by the batch you put in (e.g. index 0 was the first batch
        // you ran network->update on, index 1 is the second, etc.)
        // The inner vector is a vector of tkdnn boxes, representing the bounding boxes per image in the batch. 
        // For example: box[0][0] is batch #0's bounding box for image 1 out of 4
        std::vector<std::vector<tk::dnn::box>> boxes = network->getRawBoundingBoxes();

        // test printout for number of boxes
        std::cout << "Number of BBs: " << boxes[0].size() << std::endl;
        for (int i = 0; i < boxes[0].size(); i++) {
            std::cout << "Box " << i << " contents: \n";
            boxes[0][i].printUnscaled();
        }
    }

    // As tkDNN stands now, as you do more inference you keep adding to network->batchDetected. To prevent a memory leak, you will
    // need to periodically clear out this vector 
    network->batchDetected.clear();

    // std::cout << "Inferred in " << *network->stats.begin() << " milliseconds\n";
    dumpPerformanceStats(network);
}

#endif /* TKDNN_WRAPPER*/

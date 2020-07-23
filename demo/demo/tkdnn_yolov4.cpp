#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);

	extern char *optarg;
	extern int optind;

    std::string net = "";
    std::string input = "";
    char* ntype = "x";
    int n_classes = -1;
    int n_batch = 0;
    int c;
    // Path to TensorRT .rt runtime file
    // Path to image or video input
    // Flaggable NN architecture type (Yolo, ResNET, etc.)
    // Flaggable Batch Size (0 to 8)
    // Flaggable number of classes (1 to any)
    // Flag to squelch GUI previewing of result
    static char usage[] = "usage: %s -r network -i image -n network_type -b batch_size -c num_classes -s hide_gui\n";

    while((c = getopt(argc, argv, "r:i:nbcs")) != -1){
        switch(c) {
            case 'r':
                net = optarg;
                break;
            case 'i':
                input = optarg;
                break;
            case 'n':
                ntype = optarg;
                break;
            case 'b':
                n_batch = atoi(optarg);
                if(n_batch < 1 || n_batch > 64){
                    FatalError("FatalError: Inference Batch Size invalid (batch size must be greater than 0, but less than 64)");
                }
                break;
            case 'c':
                n_classes = atoi(optarg);
                if(n_classes < 1){
                    FatalError("FatalError: Number of classes < 1");
                }
                break;
            case 's':
                SAVE_RESULT = true;
                std::cout<<"GUI preview disabled, output will be saved to disk\n";
                break;
        }
    }

    // Check program inputs
    if (net.empty()) {	/* -n flag is mandatory, need path to NN */
		fprintf(stderr, "%s: missing -r option, path to net mandatory\n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
	} else if(input.empty()){
        fprintf(stderr, "%s: missing -i option, path to image/video mandatory\n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
    } else if (ntype == "x") {
        fprintf(stderr, "%s: missing -n option, neural network type option is mandatory\n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
    } else if ((optind+6) > argc) {	
		/* need at least one argument (change +1 to +2 for two, etc. as needeed) */

		printf("optind = %d, argc=%d\n", optind, argc);
		fprintf(stderr, "%s: Not enough arguments to run tkDNN\n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
	}

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  

    tk::dnn::DetectionNN *detNN;  

    if(ntype == "yolo"){
        detNN = &yolo;
    } else if (ntype == "centernet") {
        detNN = &cnet;
    } else if (ntype == "mbnet") {
        detNN = &mbnet;
        n_classes++;
    } else {
        FatalError("Network type invalid, allowed networks: \"yolo\", \"centernet\", \"mbnet\" \n");
    }

    std::cout<<"DEBUG: input flags valid, proceeding..." << std::endl;

    detNN->init(net, n_classes, n_batch);

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    // Stopped here: Figure out how to capture the details of an input image. 

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if(!SAVE_RESULT)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();
        
        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame; 
            if(!frame.data) 
                break;
            
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        } 
        if(!frame.data) 
            break;
    
        //inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw(batch_frame);

        if(!SAVE_RESULT){
            for(int bi=0; bi< n_batch; ++bi){
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }
        if(n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}


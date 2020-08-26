#ifndef DETECTIONNN_H
#define DETECTIONNN_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>    
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"

#define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif


namespace tk { namespace dnn {

class DetectionNN {

    protected:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_d;

        std::vector<cv::Size> originalSize;

        cv::Scalar colors[256];

        int nBatches = 1;

#ifdef OPENCV_CUDACONTRIB
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
#else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        dnnType *input;
#endif

        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         * @param bi batch index
         */
        virtual void preprocess(cv::Mat &frame, const int bi=0) = 0;

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * boundig boxes. 
         * 
         * @param bi batch index
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        virtual void postprocess(const int bi=0,const bool mAP=false) = 0;
        virtual void postprocess(const int bi,const bool mAP, const float &conf_thres, const float &nms_thres) = 0;

    public:
        int classes = 0;
        float confThreshold = 0.3; /*threshold on the confidence of the boxes*/

        std::vector<tk::dnn::box> detected; /*bounding boxes in output*/
        std::vector<std::vector<tk::dnn::box>> batchDetected; /*bounding boxes in output*/
        std::vector<double> stats; /*keeps track of inference times (ms)*/
        std::vector<std::string> classesNames;

        DetectionNN() {};
        ~DetectionNN(){};

        /**
         * Method used to inialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param tensor_path path to the rt file og the NN.
         * @param n_classes number of classes for the given dataset.
         * @param n_batches maximum number of batches to use in inference
         * @return true if everything is correct, false otherwise.
         */
        virtual bool init(const std::string& tensor_path, const int n_classes=80, const int n_batches=1, const float conf_thresh=0.3) = 0;
        
        /**
         * This method performs the whole detection of the NN.
         * 
         * @param frames frames to run detection on.
         * @param cur_batches number of batches to use in inference
         * @param save_times if set to true, preprocess, inference and postprocess times 
         *        are saved on a csv file, otherwise not.
         * @param times pointer to the output stream where to write times
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        void update(std::vector<cv::Mat>& frames, const int cur_batches=1, bool save_times=false, std::ofstream *times=nullptr, const bool mAP=false){
            if(save_times && times==nullptr)
                FatalError("save_times set to true, but no valid ofstream given");
            if(cur_batches > nBatches)
                FatalError("A batch size greater than nBatches cannot be used");
            
            //
            clock_t start_time = clock();
            std::cout << "cur_batches: " << cur_batches << std::endl;
            
            originalSize.clear();
            if(TKDNN_VERBOSE) printCenteredTitle(" TENSORRT detection ", '=', 30); 
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi){
                    if(!frames[bi].data)
                        FatalError("No image data feed to detection");
                    originalSize.push_back(frames[bi].size());
                    preprocess(frames[bi], bi);    
                }
                TKDNN_TSTOP
                if(save_times) *times<<t_ns<<";";
            }

            std::cout << "preprocess time: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;
       
            start_time = clock();

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            dim.n = cur_batches;
            {
                if(TKDNN_VERBOSE) dim.print();
                TKDNN_TSTART
                netRT->infer(dim, input_d);
                TKDNN_TSTOP
                if(TKDNN_VERBOSE) dim.print();
                stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<";";
            }

            std::cout << "infer time: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;

            start_time = clock();

            batchDetected.clear();
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi)
                    postprocess(bi, mAP);
                TKDNN_TSTOP
                if(save_times) *times<<t_ns<<"\n";
            }

            std::cout << "postprocess time: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;

        }      

        /**
         * Method to draw boundixg boxes and labels on a frame.
         * 
         * @param frames orginal frame to draw bounding box on.
         */
        void draw(std::vector<cv::Mat>& frames) {
            tk::dnn::box b;
            int x0, w, x1, y0, h, y1;
            int objClass;
            std::string det_class;

            int baseline = 0;
            float font_scale = 0.5;
            int thickness = 2;   

            for(int bi=0; bi<frames.size(); ++bi){
                // draw dets
                for(int i=0; i<batchDetected[bi].size(); i++) { 
                    b           = batchDetected[bi][i];
                    x0   		= b.x;
                    x1   		= b.x + b.w;
                    y0   		= b.y;
                    y1   		= b.y + b.h;
                    det_class 	= classesNames[b.cl];

                    // draw rectangle
                    cv::rectangle(frames[bi], cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 

                    // draw label
                    cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
                    cv::rectangle(frames[bi], cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);                      
                    cv::putText(frames[bi], det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
                }
            }
        }

        /**
         */
        void find_conf_and_nms(const int &cam_id, 
                               const std::map<int, float> &map_id_conf,
                               const std::map<int, float> &map_id_nms, 
                               float &conf_thres, 
                               float &nms_thres)
        {
            std::map<int, float>::const_iterator it;

            // conf
            it = map_id_conf.find(cam_id);
            if (it == map_id_conf.end())
            {
                conf_thres = 0.3;
            }
            else
            {
                conf_thres = it->second;
            }
            
            // nms
            it = map_id_nms.find(cam_id);
            if (it == map_id_nms.end())
            {
                nms_thres = 0.45;
            }
            else
            {
                nms_thres = it->second;
            }
        }

        /**
         * This method performs the whole detection of the NN.
         * 
         * @param frames frames to run detection on.
         * @param cur_batches number of batches to use in inference
         * @param save_times if set to true, preprocess, inference and postprocess times 
         *        are saved on a csv file, otherwise not.
         * @param times pointer to the output stream where to write times
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        void update(std::vector<cv::Mat>& frames, 
                    const int cur_batches, 
                    const std::vector<int> &cam_ids,
                    const std::map<int, float> &map_id_conf,
                    const std::map<int, float> &map_id_nms,
                    bool save_times=false, 
                    std::ofstream *times=nullptr, 
                    const bool mAP=false){
            if(save_times && times==nullptr)
                FatalError("save_times set to true, but no valid ofstream given");
            if(cur_batches > nBatches)
                FatalError("A batch size greater than nBatches cannot be used");
            
            //
            clock_t start_time = clock();
            std::cout << "cur_batches: " << cur_batches << std::endl;
            
            originalSize.clear();
            if(TKDNN_VERBOSE) printCenteredTitle(" TENSORRT detection ", '=', 30); 
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi){
                    if(!frames[bi].data)
                        FatalError("No image data feed to detection");
                    originalSize.push_back(frames[bi].size());
                    preprocess(frames[bi], bi);    
                }
                TKDNN_TSTOP
                if(save_times) *times<<t_ns<<";";
            }

            std::cout << "preprocess time: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;
       
            start_time = clock();

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            dim.n = cur_batches;
            {
                if(TKDNN_VERBOSE) dim.print();
                TKDNN_TSTART
                netRT->infer(dim, input_d);
                TKDNN_TSTOP
                if(TKDNN_VERBOSE) dim.print();
                stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<";";
            }

            std::cout << "infer time: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;

            start_time = clock();

            batchDetected.clear();
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi)
                {
                    float conf_thres = 0.3;
                    float nms_thres = 0.45;
                    find_conf_and_nms(cam_ids[bi], map_id_conf, map_id_nms, conf_thres, nms_thres);

                    postprocess(bi, mAP, conf_thres, nms_thres);
                }
                TKDNN_TSTOP
                if(save_times) *times<<t_ns<<"\n";
            }

            std::cout << "postprocess time: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;

        }      

        /**
         * Method to get boundixg boxes and labels on a frame.
         * 
         * @param frames orginal frame to draw bounding box on.
         */
        void output(const int &cur_batches,  
                    std::vector<std::vector<cv::Rect> > &vec_rects_,
                    std::vector<std::vector<int> > &vec_classes_,
                    std::vector<std::vector<float> > &vec_probs_) {
            tk::dnn::box b;
            int x0, w, x1, y0, h, y1;
            float prob;
            int objClass;
            std::string det_class;

            for(int bi=0; bi<cur_batches; ++bi){
                std::vector<cv::Rect> rects;
                std::vector<int> classes;
                std::vector<float> probs;

                for(int i=0; i<batchDetected[bi].size(); i++) { 
                    b           = batchDetected[bi][i];
                    x0   		= b.x;
                    x1   		= b.x + b.w;
                    y0   		= b.y;
                    y1   		= b.y + b.h;
                    objClass    = b.cl;
                    prob        = b.prob;

                    cv::Rect rect(x0, y0, b.w, b.h);
                    rects.push_back(rect);

                    classes.push_back(objClass);
                    probs.push_back(prob);
                }

                vec_rects_.push_back(rects);
                vec_classes_.push_back(classes);
                vec_probs_.push_back(probs);
            }
        }

};

}}

#endif /* DETECTIONNN_H*/

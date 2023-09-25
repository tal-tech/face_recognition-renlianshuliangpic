
#include "det_face_mtcnn.hpp"
#include "det_face_id.hpp"
#include<opencv2/imgproc/imgproc_c.h>

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <vector>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/progress.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>


using namespace facethink;

//设置log格式
void setupLog(std::string filename) {
    typedef boost::log::sinks::synchronous_sink< boost::log::sinks::text_file_backend > sink_t;
    boost::property_tree::ptree pt;
    boost::shared_ptr< sink_t > file_sink = boost::log::add_file_log
            (
                    boost::log::keywords::auto_flush = true,
                    boost::log::keywords::file_name = filename,
                    boost::log::keywords::format = "[%TimeStamp%]: %Message%"
            );
    boost::log::add_common_attributes();
    int log_level = pt.get<int>("log_level", 0);
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_level);
}

//遍历目录内文件或文件夹，非循环迭代
std::vector<std::string>getFilePath(std::string folder_path) {
    namespace fs = boost::filesystem;
    fs::directory_iterator end;
    int file_count = 0;
    std::vector<std::string>filePaths;
    for (fs::directory_iterator dir(folder_path); dir != end; dir++)
    {
        std::string fn = dir->path().string();
        //std::cout << fn << std::endl;
        filePaths.push_back(fn);
    }
    return filePaths;
}

//分割字符,获取图片名作为注册ID
int segmentString(std::string str) {
    std::vector<std::string> dst;
    boost::split(dst, str, boost::is_any_of("/."), boost::token_compress_on);
    //for (size_t i = 0; i< dst.size(); i++){std::cout << dst[i] << std::endl;}
    if (dst.size() == 0) { return -1; }
    while (dst.back().compare("jpg") == 0) //jpg
    {
        dst.pop_back();
    }
    std::string img_name = dst.back();
    int idx = boost::lexical_cast<int>(img_name);
    return idx;
}


int main(int argc, char *argv[]){

    setupLog("./log/det_face_id_test_ges_nomeg_1.log"); //设置日志格式
    const std::string det_model = "../det_face_id/model/";
	const std::string config_file_path = "../det_face_id/model/config.ini";
    const std::string test_imgs_folder = "../det_face_id/images/testing/";

    FaceDetectionMTCNN *face_detector = FaceDetectionMTCNN::create(
            det_model + "det_face_mtcnn_1_v2.0.0.bin",
            det_model + "det_face_mtcnn_2_v2.0.0.bin",
            det_model + "det_face_mtcnn_3_v2.0.0.bin",
            config_file_path);

	FaceIDDetection *faceID_detector = FaceIDDetection::create(
            det_model + "det_face_id_v1.0.3-1.trt",
			config_file_path);
    // Detection process
    std::vector<std::string>test_folders = getFilePath(test_imgs_folder);
    for (int i = 0; i < test_folders.size(); i++)
    {
        std::string img_path = test_folders[i];
        std::cout << "det image " << img_path << std::endl;
        cv::Mat img = cv::imread(img_path);
        std::vector<cv::Rect> rectangles;
        std::vector<float> confidences;
        std::vector<std::vector<cv::Point>> alignment;
        
        face_detector->detection_SCALE(img, rectangles, confidences, alignment);

        std::cout <<  std::to_string(i) + ":" << confidences.size() << std::endl;
    }

    return 0;
}

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
class Image {
public:
    int m_width = 0, m_height = 0;
    cv::Mat m_image;
    
	Image() = default;
	Image(const std::string& fPath) {
        std::string ext;

        int pos = (int)fPath.find(".");
        
        if (pos != std::string::npos)
            ext = fPath.substr(pos + 1, fPath.length());
        
        if (checkFormat(ext)) {
            m_image = cv::imread(fPath, 0);
            m_image.convertTo(m_image, CV_32FC1, 1.0 / 255.0f, 0);
            m_width = m_image.cols;
            m_height = m_image.rows;
        }
	};
private:

    void read_tiff(const std::string& fPath ) {

    }

    bool checkFormat(const std::string& t_format)
    {
        if (t_format == "tiff") {
            return 0;
        }
        return -1;
    }


};
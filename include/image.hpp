#ifndef IMAGE_H
#define IMAGE_H

#include "libs.hpp"

class Image {
public:
    cv::Mat m_image, m_image_og;

    int     m_width = 0;
    int     m_height = 0;
    

	Image() = default;

    Image(const std::string& fPath) {
        std::string ext;

        int pos = (int)fPath.find(".");
        
        if (pos != std::string::npos)
            ext = fPath.substr(pos + 1, fPath.length());
        
        if (checkFormat(ext)) {
            m_image_og = cv::imread(fPath, 0);
            m_image_og.convertTo(m_image, CV_32FC1, 1.0 / 255.0f, 0);

            cv::resize(m_image, m_image, cv::Size(1000, 1000));
            cv::resize(m_image_og, m_image_og, cv::Size(1000, 1000));

            m_width = m_image.cols;
            m_height = m_image.rows;
        }
	};



private:

    void read_tiff(const std::string& fPath) {};

    bool checkFormat(const std::string& t_format)
    {
        if (t_format == "tiff") {
            return 0;
        }
        return -1;
    };


};

#endif IMAGE_H
// Video FX nadstavba pro OpenCV
// 2021-10-13

#ifndef VFX_HPP_INCLUDED
#define VFX_HPP_INCLUDED

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace VFX
{
    Mat InterpolateFrame(Mat a, Mat b, int vzorek, int iterace, int poly, int pyr);
    void AnamorficFlare(Mat &frame, int width, int thresh);
    Mat lighten(Mat a, Mat b);
    Mat AlphaBlend(Mat foreground, Mat background, Mat mask, Point position);
    Mat overlay(Mat a, Mat b);
    void hdr(const Mat &image, Mat &result);

    Mat InterpolateFrame(Mat a, Mat b, int vzorek, int iterace, int poly, int pyr)
    {
        /** \brief
         *
         * \param Mat a -> first source frame
         * \param Mat b -> second source frame
         * \param int vzorek -> window sample size (15)
         * \param int iterace -> count of iteration (2)
         * \param int poly -> poly size (5)
         * \param int pyr -> poly size (25)
         * \return
         *
         */

        #define CLAMP(x,min,max) (  ((x) < (min)) ? (min) : ( ((x) > (max)) ? (max) : (x) )  )
        Mat prevframe = a.clone();
        Mat frame = b.clone();

        Mat prevgray, gray;
        Mat fflow,bflow;
        Mat flowf = prevframe.clone();
        Mat flowb = frame.clone();
        Mat final(frame.rows,frame.cols ,CV_8UC3);
        int fx,fy,bx,by;

        cvtColor(prevframe,prevgray,COLOR_BGR2GRAY);  // Convert to gray space for optical flow calculation
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        calcOpticalFlowFarneback(prevgray, gray, fflow, 0.5, pyr, vzorek, iterace, poly, 1.2, 0);  // forward optical flow
        calcOpticalFlowFarneback(gray, prevgray, bflow, 0.5, pyr, vzorek, iterace, poly, 1.2, 0);   //backward optical flow

        for (int y=0; y<frame.rows; y++)
        {
            for (int x=0; x<frame.cols; x++)
            {
                const Point2f fxy = fflow.at<Point2f>(y,x);
                fy = CLAMP(y+fxy.y*0.5,0,frame.rows);
                fx = CLAMP(x+fxy.x*0.5,0,frame.cols);

                flowf.at<Vec3b>(fy,fx) = prevframe.at<Vec3b>(y,x);

                const Point2f bxy = bflow.at<Point2f>(y,x);
                by = CLAMP(y+bxy.y*(1-0.5),0,frame.rows);
                bx = CLAMP(x+bxy.x*(1-0.5),0,frame.cols);

                flowb.at<Vec3b>(by,bx) = frame.at<Vec3b>(y,x);
            }
        }

        final = flowf*(1-0.5) + flowb*0.5;  //combination of frwd and bckward martrix
        return final;
    }

    void AnamorficFlare(Mat &frame, int width, int thresh)
    {
        /** \brief
         *
         * \param Mat &frame -> I/O processing image
         * \param int width -> image effect width
         * \param int thresh -> light threshold
         * \return
         *
         */

        Mat gray, bw, dst;
        cvtColor(frame,gray,COLOR_BGR2GRAY);
        threshold(gray,bw,thresh,255,0);
        cvtColor(bw,bw,COLOR_GRAY2BGR);
        blur(bw,bw,Size(width,1));

        for(int y = 0; y < bw.rows; y++)
        {
            for(int x = 0; x < bw.cols; x++)
            {
                Vec3b pixel = bw.at<Vec3b>(y,x);
                pixel[1] = 0;
                pixel[2] = 0;

                bw.at<Vec3b>(y,x) = pixel;
            }
        }
        add(frame,bw,dst);
        frame = dst.clone();
    }


    Mat lighten(Mat a, Mat b)
    {
        /** \brief
         *
         * \param Mat a -> image 1
         * \param Mat b -> image 2
         * \return cv::Mat
         *
         */

        Mat result = a.clone();
        Vec3b color_dest;
        for(int y = 0; y < a.rows; y++)
        {
            for (int x = 0; x < a.cols; x++)
            {
                // do it
                Vec3b intensity_a = a.at<Vec3b>(y, x);
                Vec3b intensity_b = b.at<Vec3b>(y, x);
                int blue_a = intensity_a.val[0];
                int green_a = intensity_a.val[1];
                int red_a = intensity_a.val[2];

                int blue_b = intensity_b.val[0];
                int green_b = intensity_b.val[1];
                int red_b = intensity_b.val[2];

                if((blue_a + green_a + red_a) > (blue_b + green_b + red_b))
                {
                    color_dest.val[0] = blue_a;
                    color_dest.val[1] = green_a;
                    color_dest.val[2] = red_a;
                }else
                {
                    color_dest.val[0] = blue_b;
                    color_dest.val[1] = green_b;
                    color_dest.val[2] = red_b;
                }

                result.at<Vec3b>(Point(x,y)) = color_dest;
            }
        }
        return result;
    }

    Mat AlphaBlend(Mat foreground, Mat background, Mat mask, Point position)
    {
        Mat a = foreground.clone();
        Mat b = background.clone();

        Mat img(background.size(), CV_8UC3, Scalar(0,0,0));  // foreground
        Mat maska(background.size(), CV_8UC3, Scalar(0,0,0)); // mask

        // insert a pozicování alfa obrázku
        mask.copyTo(maska(Rect(position.x,position.y,mask.cols,mask.rows)));
        a.copyTo(img(Rect(position.x,position.y,a.cols,a.rows)));

        // datové konverze
        img.convertTo(img,CV_32FC3);
        b.convertTo(b,CV_32FC3);
        maska.convertTo(maska,CV_32FC3,1.0/255);

        Mat ouImage = Mat::zeros(img.size(), img.type());
        multiply(maska, img, img);
        multiply(Scalar::all(1.0)-maska, b, b);
        add(img, b, ouImage);

        Mat x = ouImage;
        x.convertTo(x,CV_8UC3);
        return x;
    }

    Mat overlay(Mat a, Mat b)
    {
        Mat tmp_hsv;
        cvtColor(a,tmp_hsv,COLOR_BGR2HSV);

        Mat img1 = tmp_hsv.clone();
        Mat img2 = b.clone();
        Mat result = a.clone();


        for(int i = 0; i < img1.size().height; ++i)
        {
            for(int j = 0; j < img1.size().width; ++j)
            {
                //float target = float(img1.at<uchar>(i, j)) / 255;
                Vec3b target_intensity = img1.at<Vec3b>(i, j);
                float target_b = float(target_intensity.val[0])/255;
                float target_g = float(target_intensity.val[1])/255;
                float target_r = float(target_intensity.val[2])/255;

                //float blend = float(img2.at<uchar>(i, j)) / 255;
                Vec3b blend_intensity = img2.at<Vec3b>(i, j);
                float blend_r = float(blend_intensity.val[2])/255;

                // overlay
                if(target_r > 0.5)
                {
                    //result.at<float>(i, j) = (1 - (1-2*(target-0.5)) * (1-blend));
                    Vec3b color;
                    color.val[0] = target_b*255;
                    color.val[1] = target_g*255;
                    color.val[2] = (1 - (1-2*(target_r-0.5)) * (1-blend_r))*255;
                    result.at<Vec3b>(i, j) = color;
                }
                else
                {
                    //result.at<float>(i, j) = ((2*target) * blend);
                    Vec3b color;
                    color.val[0] = target_b*255;
                    color.val[1] = target_g*255;
                    color.val[2] = ((2*target_r) * blend_r)*255;
                    result.at<Vec3b>(i, j) = color;
                }
            }
        }
        // zpětný převod na RGB model
        cvtColor(result,result,COLOR_HSV2BGR);
        return result;
    }


    void hdr(const Mat &image, Mat &result)
    {
        int s = 225;
        Mat frame = image.clone();
        Mat gray = frame.clone();
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        cvtColor(gray, gray, COLOR_GRAY2BGR);
        bitwise_not(gray,gray);
        GaussianBlur(gray,gray,Size(s,s),0); // 175

        result = overlay(frame,gray);
    }






}


#endif // VFX_HPP_INCLUDED

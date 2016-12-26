#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>      //for imshow
#include <opencv2/calib3d.hpp>      //for imshow
#include <opencv2/xfeatures2d.hpp>      //for imshow
#include <opencv2/imgproc.hpp>

#include <vector>
#include <iostream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <algorithm>


using namespace std;
using namespace cv;
//#include <vector>

//RobustMatcher class taken from OpenCV2 Computer Vision Application Programming Cookbook Ch 9
class RobustMatcher {
  private:
     // pointer to the feature point detector object
     //cv::Ptr<cv::FeatureDetector> detector;
     // pointer to the feature descriptor extractor object
     //cv::Ptr<cv::DescriptorExtractor> extractor;
     // pointer to the matcher object
     //cv::Ptr<cv::DescriptorMatcher > matcher;
     float ratio; // max ratio between 1st and 2nd NN
     bool refineF; // if true will refine the F matrix
     double distance; // min distance to epipolar
     double confidence; // confidence level (probability)
  public:
     RobustMatcher() : ratio(0.95f), refineF(false),
                       confidence(0.99), distance(10.0) {

     }


  void setConfidenceLevel(
         double conf) {
     confidence= conf;
  }
  //Set MinDistanceToEpipolar
  void setMinDistanceToEpipolar(
         double dist) {
     distance= dist;
  }
  //Set ratio
  void setRatio(
         float rat) {
     ratio= rat;
  }

  // Clear matches for which NN ratio is > than threshold
  // return the number of removed points
  // (corresponding entries being cleared,
  // i.e. size will be 0)
  int ratioTest(std::vector<std::vector<cv::DMatch> >
                                               &matches) {
    int removed=0;
      // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator
             matchIterator= matches.begin();
         matchIterator!= matches.end(); ++matchIterator) {
           // if 2 NN has been identified
           if (matchIterator->size() > 1) {
               // check distance ratio
               if ((*matchIterator)[0].distance/
                   (*matchIterator)[1].distance > ratio) {
                  matchIterator->clear(); // remove match
                  removed++;
               }
           } else { // does not have 2 neighbours
               matchIterator->clear(); // remove match
               removed++;
           }
    }
    return removed;
  }

  // Insert symmetrical matches in symMatches vector
  void symmetryTest(
      const std::vector<std::vector<cv::DMatch> >& matches1,
      const std::vector<std::vector<cv::DMatch> >& matches2,
      std::vector<cv::DMatch>& symMatches) {
    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch> >::
             const_iterator matchIterator1= matches1.begin();
         matchIterator1!= matches1.end(); ++matchIterator1) {
       // ignore deleted matches
       if (matchIterator1->size() < 2)
           continue;
       // for all matches image 2 -> image 1
       for (std::vector<std::vector<cv::DMatch> >::
          const_iterator matchIterator2= matches2.begin();
           matchIterator2!= matches2.end();
           ++matchIterator2) {
           // ignore deleted matches
           if (matchIterator2->size() < 2)
              continue;
           // Match symmetry test
           if ((*matchIterator1)[0].queryIdx ==
               (*matchIterator2)[0].trainIdx &&
               (*matchIterator2)[0].queryIdx ==
               (*matchIterator1)[0].trainIdx) {

               // add symmetrical match
                 symMatches.push_back(
                   cv::DMatch((*matchIterator1)[0].queryIdx,
                             (*matchIterator1)[0].trainIdx,
                             (*matchIterator1)[0].distance));
                 break; // next match in image 1 -> image 2
           }
       }
    }
  }

  // Identify good matches using RANSAC
  // Return fundemental matrix
  cv::Mat ransacTest(
      const std::vector<cv::DMatch>& matches,
      const std::vector<cv::KeyPoint>& keypoints1,
      const std::vector<cv::KeyPoint>& keypoints2,
      std::vector<cv::DMatch>& outMatches) {
   // Convert keypoints into Point2f
   std::vector<cv::Point2f> points1, points2;
   cv::Mat fundemental;
   for (std::vector<cv::DMatch>::
         const_iterator it= matches.begin();
       it!= matches.end(); ++it) {
       // Get the position of left keypoints
       float x= keypoints1[it->queryIdx].pt.x;
       float y= keypoints1[it->queryIdx].pt.y;
       points1.push_back(cv::Point2f(x,y));
       // Get the position of right keypoints
       x= keypoints2[it->trainIdx].pt.x;
       y= keypoints2[it->trainIdx].pt.y;
       points2.push_back(cv::Point2f(x,y));
    }
   // Compute F matrix using RANSAC
   std::vector<uchar> inliers(points1.size(),0);
   if (points1.size()>0&&points2.size()>0){
      cv::Mat fundemental= cv::findFundamentalMat(
         cv::Mat(points1),cv::Mat(points2), // matching points
          inliers,       // match status (inlier or outlier)
          CV_FM_RANSAC, // RANSAC method
          distance,      // distance to epipolar line
          confidence); // confidence probability
      // extract the surviving (inliers) matches
      std::vector<uchar>::const_iterator
                         itIn= inliers.begin();
      std::vector<cv::DMatch>::const_iterator
                         itM= matches.begin();
      // for all matches
      for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
         if (*itIn) { // it is a valid match
             outMatches.push_back(*itM);
          }
       }
       if (refineF) {
       // The F matrix will be recomputed with
       // all accepted matches
          // Convert keypoints into Point2f
          // for final F computation
          points1.clear();
          points2.clear();
          for (std::vector<cv::DMatch>::
                 const_iterator it= outMatches.begin();
              it!= outMatches.end(); ++it) {
              // Get the position of left keypoints
              float x= keypoints1[it->queryIdx].pt.x;
              float y= keypoints1[it->queryIdx].pt.y;
              points1.push_back(cv::Point2f(x,y));
              // Get the position of right keypoints
              x= keypoints2[it->trainIdx].pt.x;
              y= keypoints2[it->trainIdx].pt.y;
              points2.push_back(cv::Point2f(x,y));
          }
          // Compute 8-point F from all accepted matches
          if (points1.size()>0&&points2.size()>0){
             fundemental= cv::findFundamentalMat(
                cv::Mat(points1),cv::Mat(points2), // matches
                CV_FM_8POINT); // 8-point method
          }
       }
    }
    return fundemental;
  }

  // Match feature points using symmetry test and RANSAC
  // returns fundemental matrix
  cv::Mat match(Ptr<ORB>& orb,
		  	  	cv::Mat& image1,
                cv::Mat& image2, // input images
				cv::Mat& descriptors1,
     // output matches and keypoints
     std::vector<cv::DMatch>& matches,
     std::vector<cv::KeyPoint>& keypoints1,
     std::vector<cv::KeyPoint>& keypoints2) {

   cv::Mat descriptors2;

   orb->detectAndCompute(image2, noArray(), keypoints2, descriptors2);

   // 2. Match the two image descriptors
   // Construction of the matcher
   //cv::BruteForceMatcher<cv::L2<float>> matcher;
   // from image 1 to image 2
   // based on k nearest neighbours (with k=2)
   BFMatcher matcher(NORM_HAMMING);

   std::vector<std::vector<cv::DMatch> > matches1;
   matcher.knnMatch(descriptors1,descriptors2,
       matches1, // vector of matches (up to 2 per entry)
       2);        // return 2 nearest neighbours
    // from image 2 to image 1
    // based on k nearest neighbours (with k=2)

    std::vector<std::vector<cv::DMatch> > matches2;
    matcher.knnMatch(descriptors2,descriptors1,
       matches2, // vector of matches (up to 2 per entry)
       2);        // return 2 nearest neighbours
    // 3. Remove matches for which NN ratio is
    // > than threshold
    // clean image 1 -> image 2 matches
    int removed= ratioTest(matches1);
    // clean image 2 -> image 1 matches
    removed= ratioTest(matches2);
    // 4. Remove non-symmetrical matches
    std::vector<cv::DMatch> symMatches;
    symmetryTest(matches1,matches2,symMatches);

    const int symMatchCount = symMatches.size();
   	    	//float meanboy;
   	        Point2f point1;
   	        Point2f point2;
   	        float median;
   	     vector<float> gradientList;
		 for(size_t i = 0; i < symMatchCount; i++)
   	        {
   	            point1 = keypoints1[symMatches[i].queryIdx].pt;
   	            point2 = keypoints2[symMatches[i].trainIdx].pt;
   	        	float gradient = (point2.y - point1.y) / (point2.x - point1.x);
   	        	gradientList.push_back (fabs(gradient));
   	        	std::cout << "gradient is " << fabs(gradient) << std::endl;

   	            // do something with the best points...
   	        }
		         std::sort(gradientList.begin(), gradientList.end());
		         if(gradientList.size() % 2 == 0)
		                 median = (gradientList[gradientList.size()/2 - 1] + gradientList[gradientList.size()/2]) / 2;
		         else
		                 median = gradientList[gradientList.size()/2];

		         size_t n = gradientList.size() / 2;
		             nth_element(gradientList.begin(), gradientList.begin()+n, gradientList.end());
		    	    	std::cout << "new Median method " << gradientList[n] << std::endl;

   	    	std::cout << "No of matches by shehel: " << gradientList[35] << " size " << symMatchCount << std::endl;
   	    	std::cout << "Median" << median << std::endl;

   	// std::cout << "Sym Match count: " << "(" << point1.x << ", " << point1.y << ") ("<<point2.x<<", "<<point2.y<<")"<<std::endl;

    // 5. Validate matches using RANSAC
    cv::Mat fundemental= ransacTest(symMatches,
                keypoints1, keypoints2, matches);
    // return the found fundemental matrix

    //Alternate Step 5 - finding homography
    //std::vector<uchar> inliers(keypoints1.size(), 0);
    //cv::Mat homography = cv::findHomography(cv::Mat(keypoints1), cv::Mat(keypoints2), inliers, CV_RANSAC, 1.);

    return fundemental;
  }
};


// set parameters
int main(int argc, char *argv[]) {

	//Instantiate robust matcher

	RobustMatcher rmatcher;

	//instantiate detector, extractor, matcher



	//Load input image detect keypoints
	int64 t0 = cv::getTickCount();
	cv::Mat img1;
	std::vector<cv::KeyPoint> img1_keypoints;
	cv::Mat descriptors1;

	cv::Mat img2;

	std::vector<cv::DMatch>  matches;
	img1 = imread("./img1.jpg", IMREAD_GRAYSCALE);
	resize(img1, img1, Size(480,360));
	medianBlur(img1, img1, 5);

	Ptr<ORB> orb = ORB::create(500);

	orb->detectAndCompute(img1, noArray(), img1_keypoints, descriptors1);


	img2 = imread("./img2.jpg" , IMREAD_GRAYSCALE );
	std::vector<cv::KeyPoint> img2_keypoints;



	int j=2;
	char * filename = new char[100];
	while(img2.data)
	{
	    sprintf(filename, "./img%i.jpg",j);
	    cout <<filename <<endl;
	    img2 = imread(filename, IMREAD_GRAYSCALE );
	    if(!img2.data )
	    {
	        // no more images
	        break;
	    }

		resize(img2, img2, Size(480,360));
		//resize(img1R, img1, size(), 0, 0, INTER_NEAREST);


		medianBlur(img2, img2, 5);


	    Mat fundemental = rmatcher.match(orb, img1, img2, descriptors1, matches, img1_keypoints, img2_keypoints);


	    std::string text = "res";
	    text += std::to_string(j);
	    text += ".png";
	    	drawMatches(img1, img1_keypoints, img2, img2_keypoints, matches, fundemental);
	    	      imwrite(text, fundemental);
	    	//
	    	//          double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
	    	        cout << "ORB Matching Results" << endl;
	    	        cout << "*******************************" << endl;
	    	        cout << "# Keypoints 1:                        \t" << img1_keypoints.size() << endl;
	    	        cout << "# Keypoints 2:                        \t" << img2_keypoints.size() << endl;
	    	        cout << "# Matches:                            \t" << matches.size() << endl;
	    	//        cout << "# Inliers:                            \t" << inliers1.size() << endl;
	    	//        cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
	    	        cout << endl;

	    	matches.clear();
	    	img2_keypoints.clear();
	    	j++;
	}
	//img2 = imread("../IMG_4847.JPG", IMREAD_GRAYSCALE);
	int64 t1 = cv::getTickCount();
	double secs = (t1-t0)/cv::getTickFrequency();
	std::cout << "Times passed in seconds: " << secs << std::endl;
	//equalizeHist(img1, img1);
	//equalizeHist(img2, img2);



}


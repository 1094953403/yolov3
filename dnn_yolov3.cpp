#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::dnn;
void image_detection();
void video_detection();
float confidenceThreshold = 0.25;
String yolo_cfg = "G:/opencv_learning/yolo/yolo_v3/Project1/Project1/yolo_v3_model/yolov3.cfg";
String yolo_model = "G:/opencv_learning/yolo/yolo_v3/Project1/Project1/yolo_v3_model/yolov3.weights";

int main(int argc, char** argv)
{
	//image_detection();
	video_detection();
}

void image_detection() {
	Net net = readNetFromDarknet(yolo_cfg, yolo_model);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
		printf("output layer name : %s\n", outNames[i].c_str());
	}


	vector<string> classNamesVec;
	ifstream classNamesFile("G:/opencv_learning/yolo/yolo_v3/Project1/Project1/yolo_v3_model/object_detection_classes_yolov3.txt");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);

	}

	// º”‘ÿÕºœÒ 
	Mat frame = imread("G:/opencv_learning/yolo/yolo_v3/Project1/Project1/yolo_v3_model/pedestrian.png");
	Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
	net.setInput(inputBlob);

	// ºÏ≤‚
	std::vector<Mat> outs;
	net.forward(outs, outNames);
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	ostringstream ss;
	ss << "detection time: " << time << " ms";
	putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	for (size_t i = 0; i<outs.size(); ++i)
	{
		// Network produces output blob with a shape NxC where N is a number of
		// detected objects and C is a number of classes + 4 where the first 4
		// numbers are [center_x, center_y, width, height]
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confidenceThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		String className = classNamesVec[classIds[idx]] ;
		putText(frame, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);
		rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("YOLOv3-Detections", frame);
	waitKey(0);
	return;
}

void video_detection() {
	String yolo_cfg = "G:/opencv_learning/yolo/yolo_v3/Project1/Project1/yolo_v3_model/yolov3.cfg";
	String yolo_model = "G:/opencv_learning/yolo/yolo_v3/Project1/Project1/yolo_v3_model/yolov3.weights";
	dnn::Net net = readNetFromDarknet(yolo_cfg, yolo_model);
	if (net.empty())
	{
		printf("Could not load net...\n");
		return;
	}
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
		printf("output layer name : %s\n", outNames[i].c_str());
	}
	vector<string> classNamesVec;
	ifstream classNamesFile("G:/opencv_learning/yolo/yolo_v3/Project1/Project1/yolo_v3_model/object_detection_classes_yolov3.txt");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	VideoCapture capture(0);
	/*
	VideoCapture capture;
	capture.open("D:/vcprojects/images/fbb.avi");
	if (!capture.isOpened()) {
		printf("could not open the camera...\n");
		return;
	}
	*/
	Mat frame;
	while (capture.read(frame))
	{
		//º”‘ÿÕºœÒ
		if (frame.empty())
			if (frame.channels() == 4)
				cvtColor(frame, frame, COLOR_BGRA2BGR);
		Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
		net.setInput(inputBlob, "data");
		
	


		//Mat detectionMat = net.forward("detection_out");
		//vector<double> layersTimings;
		//double freq = getTickFrequency() / 1000;
		//double time = net.getPerfProfile(layersTimings) / freq;
		//ostringstream ss;
		//ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
		//putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));


		// ºÏ≤‚
		std::vector<Mat> outs;
		net.forward(outs, outNames);
		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		ostringstream ss;
		ss << "detection time: " << time << " ms";
		putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
		vector<Rect> boxes;
		vector<int> classIds;
		vector<float> confidences;

		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confidenceThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}

		vector<int> indices;
		NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			Rect box = boxes[idx];
			String className = classNamesVec[classIds[idx]] + std::to_string(confidences[idx]);
			putText(frame, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);
			rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
		}

		/*for (int i = 0; i < detectionMat.rows; i++)
		{
			const int probability_index = 5;
			const int probability_size = detectionMat.cols - probability_index;
			float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
			size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
			float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
			if (confidence > confidenceThreshold)
			{
				float x = detectionMat.at<float>(i, 0);
				float y = detectionMat.at<float>(i, 1);
				float width = detectionMat.at<float>(i, 2);
				float height = detectionMat.at<float>(i, 3);
				int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
				int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
				int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
				int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
				Rect object(xLeftBottom, yLeftBottom,
					xRightTop - xLeftBottom,
					yRightTop - yLeftBottom);
				rectangle(frame, object, Scalar(0, 255, 0));
				if (objectClass < classNamesVec.size())
				{
					ss.str("");
					ss << confidence;
					String conf(ss.str());
					String label = String(classNamesVec[objectClass]) + ": " + conf;
					int baseLine = 0;
					Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
					rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom),
						Size(labelSize.width, labelSize.height + baseLine)),
						Scalar(255, 255, 255), -1);
					putText(frame, label, Point(xLeftBottom, yLeftBottom + labelSize.height),
						FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
				}
			}
		}*/
		imshow("YOLOv3: Detections", frame);
		if (waitKey(1) >= 0) break;
	}
}
// VideoAnotationPreProcess.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <tchar.h>
#include <sstream>
#include <iostream>
#include <fstream>
extern "C"{
	#include <libavcodec/avcodec.h>
	#include <libavformat/avformat.h>
	#include <libswscale/swscale.h>
}
#include "opencv2/video/background_segm.hpp"
#include "opencv2/core/core.hpp""
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "tinyxml2.h"


// compatibility with newer API
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif
using namespace cv;


struct Take
{
	string name;
	int bgFrame;
	string colorPath;
	string depthpath;
	string bgPath;
};


float cameraDepthMatrix[3][3];
float cameraColorMatrix[3][3];
float rotateMatrix[3][3];
float translateMatrix[3];


AVFormatContext   *pFormatCtx = NULL;
int               i, videoStream;
AVCodecContext    *pCodecCtxOrig = NULL;
AVCodecContext    *pCodecCtx = NULL;
AVCodec           *pCodec = NULL;
AVFrame           *pFrame = NULL;
AVFrame           *pFrameRGB = NULL;
AVStream		  *pStream = NULL;
AVPacket          packet;
int               frameFinished;
int               numBytes;
uint8_t           *buffer = NULL;
struct SwsContext *sws_ctx = NULL;

string annotationsPath;
string cameraPath;
int ntakes;
vector<Take> takes;
ushort depths[217088];
Mat bgVid;
Mat bgKin;

ushort getDepthValueAt(string dpath,int x, int y){
	//Reading data
	FILE *f = fopen(dpath.c_str(), "rb");
	fread(depths, sizeof (ushort), 217088, f);
	int idx = y * 512 + x;
	return depths[idx];
	fclose(f);
}

void loadCameraParams(){
	tinyxml2::XMLDocument docc;
	docc.LoadFile(cameraPath.c_str());
	char *part;


	const char *d = docc.FirstChildElement()->FirstChildElement("Depth_intrinsics")->GetText();

	part = strtok((char *)d, " ");
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{

			cameraDepthMatrix[i][j] = atof(part);
			part = strtok(NULL, " ");
		}
	}

	const char *c = docc.FirstChildElement()->FirstChildElement("Color_intrinsics")->GetText();

	part = strtok((char *)c, " ");
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			cameraColorMatrix[i][j] = atof(part);
			part = strtok(NULL, " ");
		}
	}
}

int openFFMPEGVideo(string path){
	// Register all formats and codecs
	av_register_all();

	// Open video file
	if (avformat_open_input(&pFormatCtx, path.c_str(), NULL, NULL) != 0)
		return -1; // Couldn't open file

	// Retrieve stream information
	if (avformat_find_stream_info(pFormatCtx, NULL)<0)
		return -1; // Couldn't find stream information

	// Dump information about file onto standard error
	av_dump_format(pFormatCtx, 0, path.c_str(), 0);

	// Find the first video stream
	videoStream = -1;
	for (i = 0; i<pFormatCtx->nb_streams; i++)
	if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
		pStream = pFormatCtx->streams[i];
		videoStream = i;
		break;
	}
	if (videoStream == -1)
		return -1; // Didn't find a video stream

	// Get a pointer to the codec context for the video stream
	pCodecCtxOrig = pFormatCtx->streams[videoStream]->codec;
	// Find the decoder for the video stream
	pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);
	if (pCodec == NULL) {
		fprintf(stderr, "Unsupported codec!\n");
		return -1; // Codec not found
	}
	// Copy context
	pCodecCtx = avcodec_alloc_context3(pCodec);
	if (avcodec_copy_context(pCodecCtx, pCodecCtxOrig) != 0) {
		fprintf(stderr, "Couldn't copy codec context");
		return -1; // Error copying codec context
	}

	// Open codec
	if (avcodec_open2(pCodecCtx, pCodec, NULL)<0)
		return -1; // Could not open codec

	// Allocate video frame
	pFrame = av_frame_alloc();

	// Allocate an AVFrame structure
	pFrameRGB = av_frame_alloc();
	if (pFrameRGB == NULL)
		return -1;

	// Determine required buffer size and allocate buffer
	numBytes = avpicture_get_size(AV_PIX_FMT_RGB24, pCodecCtx->width,
		pCodecCtx->height);
	buffer = (uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

	// Assign appropriate parts of buffer to image planes in pFrameRGB
	// Note that pFrameRGB is an AVFrame, but AVFrame is a superset
	// of AVPicture
	avpicture_fill((AVPicture *)pFrameRGB, buffer, AV_PIX_FMT_RGB24,
		pCodecCtx->width, pCodecCtx->height);

	// initialize SWS context for software scaling
	sws_ctx = sws_getContext(pCodecCtx->width,
		pCodecCtx->height,
		pCodecCtx->pix_fmt,
		pCodecCtx->width,
		pCodecCtx->height,
		AV_PIX_FMT_BGR24,
		SWS_BILINEAR,
		NULL,
		NULL,
		NULL
		);

	return 0;
}

void loadParams(const char* path){
	tinyxml2::XMLDocument inputdoc;
	inputdoc.LoadFile(path);
	tinyxml2::XMLElement *head = inputdoc.FirstChildElement("input");
	annotationsPath = head->FirstChildElement("AnnotationsFile")->GetText();
	cameraPath = head->FirstChildElement("CalibrationFile")->GetText();
	tinyxml2::XMLElement *t = head->FirstChildElement("KinectTakeInfo")->FirstChildElement("Take");
	ntakes = 0;
	while (t != NULL){
		string name = t->FirstChildElement("Id")->GetText();
		int bgFrame = atoi(t->FirstChildElement("bgFrame")->GetText());
		std::cout << bgFrame;
		string colorPath = t->FirstChildElement("ColorPath")->GetText();
		string depthPath = t->FirstChildElement("DepthPath")->GetText();
		string bgpath = t->FirstChildElement("bgPath")->GetText();
		Take muhtake = { name, bgFrame,colorPath, depthPath,bgpath };
		takes.push_back(muhtake);
		t = t->NextSiblingElement();
		ntakes++;
	}
}

string getBgVideo(Take t){
	printf("requested: %d (frame)\n", t.bgFrame);
	int res = avformat_seek_file(pFormatCtx, videoStream, INT64_MIN, t.bgFrame, INT64_MAX, AVSEEK_FLAG_ANY);
	avcodec_flush_buffers(pCodecCtx);
	int nframe = 0;
	bool notdone = true;
	while (av_read_frame(pFormatCtx, &packet) >= 0) {

		// Is this a packet from the video stream?
		if (packet.stream_index == videoStream) {
			// Decode video frame
			avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);
			// Did we get a video frame?
			if (frameFinished) {

				// Convert the image from its native format to RGB
				sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data,
					pFrame->linesize, 0, pCodecCtx->height,
					pFrameRGB->data, pFrameRGB->linesize);

				cv::Mat img(pFrame->height,
					pFrame->width,
					CV_8UC3,
					pFrameRGB->data[0]);

				imwrite("vidbg.jpg", img);
				return "vidbg.jpg";
			}
		}
	}

}

void processAnnotations(tinyxml2::XMLDocument *doc, tinyxml2::XMLElement *annotation, Take t ){
	// Read frames and save first five frames to disk
	i = 0;
	tinyxml2::XMLElement *el = annotation->FirstChildElement("annotation");
	string type = el->Attribute("type");
	
	double request;
	el->FirstChildElement("begin")->QueryDoubleText(&request);
	int x;
	int y;
	if (type == "ink"){

		return;
	}
	el->FirstChildElement("position")->FirstChildElement("x")->QueryIntText(&x);
	el->FirstChildElement("position")->FirstChildElement("y")->QueryIntText(&y);
	
	//double val = av_q2d(pStream->r_frame_rate);

	//float correctedRequest = request / 100;
	//int second = (int)std::floor(correctedRequest);
	//float decimals = correctedRequest - second;
	//int frame = std::round(decimals *val);
	//correctedRequest = second +(frame / 100.0);
	
	//int64_t request_timestamp = correctedRequest * val; //* 2;
	printf("requested: %.2f (frame)\n", request);
	int res = avformat_seek_file(pFormatCtx, videoStream, INT64_MIN, request, INT64_MAX, AVSEEK_FLAG_ANY);
	avcodec_flush_buffers(pCodecCtx);
	int nframe = 0;
	bool notdone = true;
	while (av_read_frame(pFormatCtx, &packet) >= 0) {

		// Is this a packet from the video stream?
		if (packet.stream_index == videoStream) {
			// Decode video frame
			avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);
			// Did we get a video frame?
			if (frameFinished) {

				// Convert the image from its native format to RGB
				sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data,
					pFrame->linesize, 0, pCodecCtx->height,
					pFrameRGB->data, pFrameRGB->linesize);

				cv::Mat img_vid(pFrame->height,
					pFrame->width,
					CV_8UC3,
					pFrameRGB->data[0]);

				//load according frame from disk 
				double error = 0;
				int newrequest = request - error;

				int val = std::round(av_q2d(pStream->r_frame_rate));
				int seconds =(int) std::floor(newrequest / val);
				int frames = (int)newrequest % val;
				frames = std::round(frames *30.0 / val);
				
				
				bool foundfile = false;
				Mat img_kin;
				int attempts = 100;
				while (attempts > 0)
				{
					std::stringstream ss;
					ss << t.colorPath << "\\color" << seconds << "," << frames << ".bmp";

					string colorname = ss.str();

					std::ifstream ifile(colorname);
					if (ifile){
						img_kin = imread(colorname, CV_LOAD_IMAGE_COLOR);
						cv::flip(img_kin, img_kin, 1);
						break;
					}
					else{
						frames++;
						if (frames == 30){
							frames = 0;
							seconds++;
						}
					}
				}


				//-- Step 1: Detect the keypoints using SURF Detector
				int minHessian = 1500;
				StarFeatureDetector star = cv::StarDetector();
				SiftFeatureDetector sift = cv::SiftFeatureDetector();
				SurfFeatureDetector surf(minHessian);
				std::vector<KeyPoint> keypoints_video, keypoints_kin;
			

				img_kin = bgKin  - img_kin;
				cv::bitwise_not(img_kin, img_kin);
				img_vid = bgVid - img_vid;
				cv::bitwise_not(img_vid, img_vid);
			//	cv::Mat foremask;
			//	cv::BackgroundSubtractorMOG2 bg;
			//	bg.set("nmixtures", 3);
			//	bg.operator()(bgVid, foremask);
			//	bg.operator()(img_vid, foremask);

				surf.detect(img_kin, keypoints_kin);
				surf.detect(img_vid, keypoints_video);
				
			
				// computing descriptors
				SurfDescriptorExtractor extractor;
				//SiftDescriptorExtractor extractor;
				//BriefDescriptorExtractor extractor;
				Mat descriptorsvid, descriptorskin;
				extractor.compute(img_vid, keypoints_video, descriptorsvid);
				extractor.compute(img_kin, keypoints_kin, descriptorskin);


				// matching descriptors
				//BFMatcher matcher(NORM_L2);
				FlannBasedMatcher matcher;
				std::vector< DMatch > matches;
				matcher.match(descriptorsvid, descriptorskin, matches);

				double max_dist = 0; double min_dist = 100;

				//-- Quick calculation of max and min distances between keypoints
				for (int i = 0; i < descriptorsvid.rows; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				std::vector< DMatch > good_match;
				float mindist = 100000;
				float minquality = 100000;
				int minindex = 0;
				for (int i = 0; i < matches.size(); i++){
					float ptx = keypoints_video[matches[i].queryIdx].pt.x;
					float pty = keypoints_video[matches[i].queryIdx].pt.y;
					float dist = std::sqrt(std::pow(x - ptx, 2) + std::pow(y - pty, 2));

					if (matches[i].distance < min_dist * 1.5 && dist < mindist){
						minindex = i;
						mindist = dist;
					}
				}
				good_match.push_back(matches[minindex]);

				/*
				//-- Draw keypoints vid
				Mat img_keypoints_video;
				drawKeypoints(img_vid, keypoints_video, img_keypoints_video, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				imwrite("videoframe.jpg", img_keypoints_video);
				imshow("Frame", img_keypoints_video);
				waitKey(0);


				//-- Draw keypoints img
				Mat img_keypoints_kin;
				drawKeypoints(img_kin, keypoints_kin, img_keypoints_kin, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				imwrite("kinectframe.jpg", img_keypoints_kin);
				imshow("Frame", img_keypoints_kin);
				waitKey(0);

				*/
				//-- Draw matches
				Mat img_matches;
				drawMatches(img_vid, keypoints_video, img_kin, keypoints_kin, good_match, img_matches);

				imwrite("matches.jpg", img_matches);
				//-- Show detected matches

				namedWindow("Matches", WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);// 
				imshow("Matches", img_matches);
				waitKey(0);

				//Getting Z value
				Point2f position = keypoints_kin[matches[minindex].imgIdx].pt;
				position.x = 512 - position.x;
				std::stringstream ss;
				ss << t.depthpath << "\\depthdata" << seconds << "," << frames;
				ushort depth = getDepthValueAt(ss.str(), position.x, position.y);
				
				float x = (depth * (position.x - cameraDepthMatrix[0][2]) / cameraDepthMatrix[0][0]) / 1000;
				float y = -(depth * (position.y - cameraDepthMatrix[1][2]) / cameraDepthMatrix[1][1]) / 1000;
				float z = depth / 1000.0;

				//adding stuff to the xml

				tinyxml2::XMLElement *depthpos = doc->NewElement("positionKin");
				tinyxml2::XMLElement *xxml = doc->NewElement("x");
				xxml->SetText(x);
				tinyxml2::XMLElement *yxml = doc->NewElement("y");
				yxml->SetText(y);
				tinyxml2::XMLElement *zxml = doc->NewElement("z");
				zxml->SetText(z);
				depthpos->InsertFirstChild(xxml);
				depthpos->InsertEndChild(yxml);
				depthpos->InsertEndChild(zxml);
				el->InsertEndChild(depthpos);


				av_free_packet(&packet);
				break;

			}
		}
		// Free the packet that was allocated by av_read_frame
		av_free_packet(&packet);
		
	}
}

int main(int argc, const char* argv[])
{
	// Initalizing these to NULL prevents segfaults!
	
	if (argc < 2) {
		printf("Please provide a xml file\n");
		return -1;
	}
	loadParams(argv[1]);
	loadCameraParams();
	tinyxml2::XMLDocument *doc = new tinyxml2::XMLDocument();
	doc->LoadFile(annotationsPath.c_str());
	//find the take
	tinyxml2::XMLElement *head = doc->FirstChildElement("annotation_document");
	tinyxml2::XMLElement *session = head->FirstChildElement("session");
	while (session != NULL){ 
		tinyxml2::XMLElement *take = session->FirstChildElement("take"); 
		while (take != NULL){
			//find the path
			string path = take->FirstChildElement("file-name1")->GetText();
			string name = take->Attribute("name");

			Take takeobj;
			for (int i = 0; i < takes.size(); i++){
				if (takes[i].name == name){
					takeobj = takes[i];
					break;
				}
			}
			if (openFFMPEGVideo(path) < 0){
				return -1;
			}

			//look through xml
			tinyxml2::XMLElement *annotation = take->FirstChildElement("annotations")->FirstChildElement("annotation_set");

			bgKin = imread(takeobj.bgPath, CV_LOAD_IMAGE_COLOR);
			cv::flip(bgKin, bgKin, 1);
			getBgVideo(takeobj);
			bgVid = imread("vidbg.jpg", CV_LOAD_IMAGE_COLOR);



			while (annotation != NULL){
				processAnnotations(doc, annotation, takeobj);
				annotation = annotation->NextSiblingElement();
			}
			// Free the RGB image
			av_free(buffer);
			av_frame_free(&pFrameRGB);

			// Free the YUV frame
			av_frame_free(&pFrame);

			// Close the codecs
			avcodec_close(pCodecCtx);
			avcodec_close(pCodecCtxOrig);

			// Close the video file
			avformat_close_input(&pFormatCtx);
			take = take->NextSiblingElement();
		}
		session = session->NextSiblingElement();
	}
	doc->SaveFile("output.xml");
	return 0;

}


/*
int main(int argc, char *argv[]) {
	// Initalizing these to NULL prevents segfaults!
	AVFormatContext   *pFormatCtx = NULL;
	int               i, videoStream;
	AVCodecContext    *pCodecCtxOrig = NULL;
	AVCodecContext    *pCodecCtx = NULL;
	AVCodec           *pCodec = NULL;
	AVFrame           *pFrame = NULL;
	AVFrame           *pFrameRGB = NULL;
	AVPacket          packet;
	int               frameFinished;
	int               numBytes;
	uint8_t           *buffer = NULL;
	struct SwsContext *sws_ctx = NULL;


	if (argc < 3) {
		printf("Please provide a xml file and a folder\n");
		return -1;
	}

	tinyxml2::XMLDocument doc;
	tinyxml2::XMLError e = doc.LoadFile(argv[1]);

	//find the take
	tinyxml2::XMLElement *head = doc.FirstChildElement("annotation_document");
	tinyxml2::XMLElement *take = head->FirstChildElement("session")->FirstChildElement("take");

	string path = take->FirstChildElement("path1")->GetText();

	// Register all formats and codecs
	av_register_all();

	// Open video file
	if (avformat_open_input(&pFormatCtx, path.c_str(), NULL, NULL) != 0)
		return -1; // Couldn't open file

	// Retrieve stream information
	if (avformat_find_stream_info(pFormatCtx, NULL)<0)
		return -1; // Couldn't find stream information

	// Dump information about file onto standard error
	av_dump_format(pFormatCtx, 0, argv[1], 0);

	// Find the first video stream
	videoStream = -1;
	for (i = 0; i<pFormatCtx->nb_streams; i++)
	if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
		videoStream = i;
		break;
	}
	if (videoStream == -1)
		return -1; // Didn't find a video stream

	// Get a pointer to the codec context for the video stream
	pCodecCtxOrig = pFormatCtx->streams[videoStream]->codec;
	// Find the decoder for the video stream
	pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);
	if (pCodec == NULL) {
		fprintf(stderr, "Unsupported codec!\n");
		return -1; // Codec not found
	}
	// Copy context
	pCodecCtx = avcodec_alloc_context3(pCodec);
	if (avcodec_copy_context(pCodecCtx, pCodecCtxOrig) != 0) {
		fprintf(stderr, "Couldn't copy codec context");
		return -1; // Error copying codec context
	}

	// Open codec
	if (avcodec_open2(pCodecCtx, pCodec, NULL)<0)
		return -1; // Could not open codec

	// Allocate video frame
	pFrame = av_frame_alloc();

	// Allocate an AVFrame structure
	pFrameRGB = av_frame_alloc();
	if (pFrameRGB == NULL)
		return -1;

	// Determine required buffer size and allocate buffer
	numBytes = avpicture_get_size(PIX_FMT_RGB24, pCodecCtx->width,
		pCodecCtx->height);
	buffer = (uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

	// Assign appropriate parts of buffer to image planes in pFrameRGB
	// Note that pFrameRGB is an AVFrame, but AVFrame is a superset
	// of AVPicture
	avpicture_fill((AVPicture *)pFrameRGB, buffer, PIX_FMT_RGB24,
		pCodecCtx->width, pCodecCtx->height);

	// initialize SWS context for software scaling
	sws_ctx = sws_getContext(pCodecCtx->width,
		pCodecCtx->height,
		pCodecCtx->pix_fmt,
		pCodecCtx->width,
		pCodecCtx->height,
		PIX_FMT_RGB24,
		SWS_BILINEAR,
		NULL,
		NULL,
		NULL
		);

	// Read frames and save first five frames to disk
	i = 0;
	while (av_read_frame(pFormatCtx, &packet) >= 0) {
		// Is this a packet from the video stream?
		if (packet.stream_index == videoStream) {
			// Decode video frame
			avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

			// Did we get a video frame?
			if (frameFinished) {
				// Convert the image from its native format to RGB
				sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data,
					pFrame->linesize, 0, pCodecCtx->height,
					pFrameRGB->data, pFrameRGB->linesize);

				cv::Mat img_vid(pCodecCtx->height,
					pCodecCtx->width,
					CV_8UC3,
					pFrameRGB->data[0],
					pFrameRGB->linesize[0]);

				//-- Step 1: Detect the keypoints using SURF Detector
				int minHessian = 400;
				SurfFeatureDetector detector(minHessian);
				std::vector<KeyPoint> keypoints_video, keypoints_kin;
				detector.detect(img_vid, keypoints_video);
				Mat img_keypoints_video;
				drawKeypoints(img_vid, keypoints_video, img_keypoints_video, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				imwrite("videoframe.jpg", img_keypoints_video);
				imshow("Frame", img_keypoints_video);
				waitKey(0);
				// Save the frame to disk
			
			}
		}

		// Free the packet that was allocated by av_read_frame
		av_free_packet(&packet);
	}

	// Free the RGB image
	av_free(buffer);
	av_frame_free(&pFrameRGB);

	// Free the YUV frame
	av_frame_free(&pFrame);

	// Close the codecs
	avcodec_close(pCodecCtx);
	avcodec_close(pCodecCtxOrig);

	// Close the video file
	avformat_close_input(&pFormatCtx);

	return 0;
}
*/

/*
int main( int argc, char** argv )
{
if( argc != 3 )
{  return -1; }

Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );

if( !img_1.data || !img_2.data )
{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

//-- Step 1: Detect the keypoints using SURF Detector
int minHessian = 1500;

SurfFeatureDetector detector( minHessian );

std::vector<KeyPoint> keypoints_1, keypoints_2;

detector.detect( img_1, keypoints_1 );
detector.detect( img_2, keypoints_2 );

//-- Draw keypoints
Mat img_keypoints_1; Mat img_keypoints_2;

drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

// computing descriptors
SurfDescriptorExtractor extractor;
//SiftDescriptorExtractor extractor;
//BriefDescriptorExtractor extractor;
Mat descriptors1, descriptors2;
extractor.compute(img_1, keypoints_1, descriptors1);
extractor.compute(img_2, keypoints_2, descriptors2);


// matching descriptors
//BFMatcher matcher(NORM_L2);
FlannBasedMatcher matcher;
std::vector< DMatch > matches;
matcher.match(descriptors1, descriptors2, matches);

double max_dist = 0; double min_dist = 100;

//-- Quick calculation of max and min distances between keypoints
for (int i = 0; i < descriptors1.rows; i++)
{
	double dist = matches[i].distance;
	if (dist < min_dist) min_dist = dist;
	if (dist > max_dist) max_dist = dist;
}

std::vector< DMatch > good_match;
int minindex = 0;
for (int i = 0; i < matches.size(); i++){
	if (matches[i].distance < min_dist *10){
		good_match.push_back(matches[i]);
		
	}
}
good_match.push_back(matches[minindex]);


Mat img_matches;
drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_match, img_matches);

	imwrite("matches.jpg", img_matches);
//-- Show detected matches

namedWindow("Matches", WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);// 
imshow("Matches", img_matches);
waitKey(0);
//-- Show detected (drawn) keypoints
imshow("Keypoints 1", img_keypoints_1 );
imshow("Keypoints 2", img_keypoints_2 );

waitKey(0);

return 0;
}
*/
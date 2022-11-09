#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "eeg.h"
#include "Parser_Filter.h"
#include <string.h>
#include <string>
#include <fstream>
#include "imu.h"
#include "kalmanFilter.h"
#include <errno.h>
#include <csignal>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <pigpio.h>
#include <iostream>
#include "cv.h"

using namespace cv;
using namespace std;

EEG EEG;
FILE* Data_text;
FILE* data;
extern double dataArray[14];
int prevVal = 0, currVal = 0;
int elapsedS = 0, elapsedMS = 0;
int startS = 0, startMS = 0;
int currentS = 0, currentMS = 0;
double timeMS = 0.0f;

pthread_t *p1, *p2, *p3;
char eegthread[] = "EEG Thread";
char imuthread[] = "IMU Thread";
char cvthread[] = "CV Thread";

void *eegFunc(void *arg) {
	unsigned char streamByte;
	while(1) {
		if(serDataAvailable (EEG.serial_port)){
			// printf("new ser data avail: %lf\n",(double) ((double) elapsedS + timeMS) );
			streamByte = eegRead(&EEG);
			// printing streamByte for debugging purposes
			// printf("\n streamByte: %d", streamByte);
			fflush(stdout);
		}
		if(currVal % 2 == 0){
			if(prevVal != currVal){
				prevVal = currVal;
				// storing elapsed time to dataArray buffer
				dataArray[0] = (double) ((double) elapsedS + timeMS);
				// for writing for binary
				fwrite(dataArray, sizeof(double), 14, data);
				// for writing for txt file
				//fprintf(EEG.Data_Text, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",dataArray[0],dataArray[1],dataArray[2],dataArray[3],dataArray[4],dataArray[5],dataArray[6],dataArray[7],dataArray[8],dataArray[9],dataArray[10]);
				//printf("%lf\n",dataArray[0]);
			}
		}
	}
}

void *imuFunc(void *arg) {
	while(1) {
		//printf("imu\n");
	}
}

void *cvFunc(void *arg) {
	while(1) {
		//printf("cv\n");
	}
}

void signalHandler (int signum) {
    cout << "Interrupt Handling #" << signum << endl;
	serClose(EEG.serial_port);
    free(EEG.queue);
    free(EEG.queue1);
    fclose(data);
	// turning back gpio 4 and 17 to input
	gpioWrite(17,0);
	gpioSetMode(4,PI_INPUT);
	gpioSetMode(17,PI_INPUT);
	gpioStopThread(p3);
	gpioStopThread(p2);
	gpioStopThread(p1);
	gpioTerminate();
	fclose(Data_text);

    // cleanup and close up stuff here  
    // terminate program  

    exit(signum);
}


int main( int argc, char **argv ) {
	if (gpioInitialise()<0) //initialises pigpio.h
    {
        //if pigpio initialisation failed
        cout<<"pigpio.h initialisation failed\n";
        return -1;
    }

    // register SIGINT and signal handler
    signal(SIGINT, signalHandler); 
	
	// gpio 4 is set to output for LED indicator for poor signal quality
	gpioSetMode(4,PI_OUTPUT);
	gpioSetMode(4,PI_OUTPUT);
	
	Data_text = fopen("Text_Data.txt","w");
	
	data = fopen("data.bin","wb");
	
	dataArray[0]=0;		// index 0 for time
	dataArray[1]=0;		// index 1 for EEG Raw Values
	dataArray[2]=0;		// index 2 for EEG Values with MAV Filter
	dataArray[3]=0;		// index 3 for Delta Band
	dataArray[4]=0;		// index 4 for Theta Band
	dataArray[5]=0;		// index 5 for Low-alpha Band
	dataArray[6]=0;		// index 6 for High-alpha Band
	dataArray[7]=0;		// index 7 for Low-beta Band
	dataArray[8]=0;		// index 8 for High-beta Band
	dataArray[9]=0;		// index 9 for Low-gamma Band
	dataArray[10]=0;	// index 10 for Mid-gamma Band
	dataArray[11]=0;	// index 11 for Poor-Signal Quality
	dataArray[12]=0;	// index 12 for fitterRoll IMU values
	dataArray[13]=0;	// index 13 for fittedPitch IMU values
	
	p1 = gpioStartThread(eegFunc,eegthread);
	p2 = gpioStartThread(imuFunc,imuthread);
	p3 = gpioStartThread(cvFunc,cvthread);

	// getting start time
	gpioTime(0,&startS,&startMS);


	while(1){
		// getting current time
		gpioTime(0,&currentS,&currentMS);
		// getting elapsed time from start to current
		elapsedS = currentS - startS;
		elapsedMS = currentMS - startMS;
		// getting integer elapsed Microsecond and converting it to float for an actual float value of microsecond
		timeMS = elapsedMS/1000000.0f;
		// getting elapsed millisecond from elapsed microsecond
		currVal = elapsedMS/1000;
		printf("%lf\n",dataArray[0]);
	}
   
    return 0;
}

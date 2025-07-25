
.PHONY: all clean run

all:
	g++ -std=c++17 yolov8.cpp \
  		-I/usr/share/cix/include/opencv4 \
  		-I/usr/share/cix/include/npu \
  		-L/usr/share/cix/lib \
  		-lopencv_core \
  		-lopencv_imgproc \
  		-lopencv_imgcodecs \
  		-lopencv_highgui \
  		-lopencv_dnn \
  		-lopencv_dnn_objdetect \
  		-lnoe -O3 -march=native -DNDEBUG -pipe \
  		-o yolov8

clean:
	rm -f yolov8


run:
	./yolov8 build/yolov8s.cix bus.jpg 0.25 0.45
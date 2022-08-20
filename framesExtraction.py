# WARNING: put the dataset folder into the project folder for the dataset acquisition
import cv2


def testExtraction(dataVideoTest):
    dataVideoTest.cap = cv2.VideoCapture(dataVideoTest.pathToFile)
    dataVideoTest.fps = dataVideoTest.cap.get(cv2.CAP_PROP_FPS)
    dataVideoTest.standard_width = int(dataVideoTest.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dataVideoTest.standard_height = int(dataVideoTest.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dataVideoTest.numRawFrames = extraction(dataVideoTest, 1, dataVideoTest.cap)

    if dataVideoTest.cap.isOpened():
        dataVideoTest.cap.release()
        cv2.destroyAllWindows()
    dataVideoTest.cap = None


def extraction(dataVideoTest, skipFrame, cap):
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        a = count % skipFrame
        if ret and count % skipFrame == 0:
            if dataVideoTest.flip:
                frame = cv2.flip(frame, 1)  # 0 horizontally, 1 vertically, -1 both
            frame = cv2.resize(frame, (dataVideoTest.standard_width, dataVideoTest.standard_height))
            if dataVideoTest.pathRawFrames is not None:
                cv2.imwrite(dataVideoTest.pathRawFrames +
                            "frame" + str(count) + ".jpg", frame)
                print('Extracting frame: ' + str(count))
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            break
        count += 1

    return count

import ximea
print(ximea.__version__)

from ximea import xiapi
import PIL.Image

from ximea import xiapi
import cv2
import time

def on_change(value):

    cam.set_exposure(value)
# create instance for first connected camera
try:
    cam = xiapi.Camera(1)
    # start communication
    print('Opening first camera...')
    cam.open_device()
    # settings
    cam.set_exposure(200000)
    # create instance of Image to store image data and metadata
    img = xiapi.Image()
    # start data acquisition
    print('Starting data acquisition...')
    cam.start_acquisition()

except Exception as e:
    print(str(e))

try:
    print('Starting video. Press Space bar  to exit.')
    t0 = time.time()
    cv2.namedWindow('XiCAM Camera')
    cv2.createTrackbar('Exposure', 'XiCAM Camera', 0, 900000, on_change)

    while True:
        # get data and pass them from camera to img
        if cam.CAM_OPEN==True:
            cam.get_image(img)
            data = img.get_image_data_numpy()
        else:
            data = cv2.imread("xi_example.bmp")
        # create numpy array with data from camera. Dimensions of the array are
        # determined by imgdataformat


        # show acquired image with time since the beginning of acquisition
        font = cv2.FONT_HERSHEY_SIMPLEX
        #text = '{:5.2f}'.format(time.time() - t0)
        # cv2.putText(
        #     data, text, (900, 150), font, 4, (255, 255, 255), 2
        # )
        #2176 4112

        data=cv2.resize(data, (1000, 700))
        cv2.rectangle(data, (400, 100), (800, 600), (9, 95, 0), 15)
        cv2.imshow('XiCAM Camera', data)
        if cv2.waitKey(1) == ord(' '):
            break

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print('Stopping acquisition...')
    cam.stop_acquisition()

    # stop communication
    cam.close_device()

    print('Done.')

# stop data acquisition

#
# # create instance for first connected camera
#
# cam = xiapi.Camera()
#
# # start communication
# print('Opening first camera...')
# cam.open_device()
#
# # settings
# cam.set_imgdataformat('XI_MONO8')
# cam.set_exposure(200000)
#
# # create instance of Image to store image data and metadata
# img = xiapi.Image()
#
# # start data acquisition
# print('Starting data acquisition...')
# cam.start_acquisition()
#
# # get data and pass them from camera to img
# cam.get_image(img)
# data = img.get_image_data_numpy()
#
# # stop data acquisition
# print('Stopping acquisition...')
# cam.stop_acquisition()
#
# # stop communication
# cam.close_device()
#
# # save acquired image
# print('Saving image...')
# img = PIL.Image.fromarray(data, 'L')
# img.save('xi_example.bmp')
# # img.show()
#
# print('Done.')
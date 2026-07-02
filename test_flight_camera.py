import airsim
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToPositionAsync(10, 0, -10, 5).join()

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
])
img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
img = img.reshape(responses[0].height, responses[0].width, 3)
cv2.imwrite(r"c:\DroneProject\camera_test.png", img)
print("Saved c:\\DroneProject\\camera_test.png", img.shape)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
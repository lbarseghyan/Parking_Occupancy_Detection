# Real-Time Parking Occupancy Detection System

Nowadays, finding vacant parking spaces in urban areas has become a problem. Drivers spend a lot of time finding vacant parking spaces, which is stressful and results in traffic congestion and increased fuel consumption. To solve this problem, we developed a deep learning-based system that can accurately detect and classify parking spaces as occupied or unoccupied in real-time, using live video feeds from cameras installed in the parking lot. The suggested system consists of two components: object detection using YOLO and parking space occupancy detection using IoU (intersection over the union). To train and test our object detection model , we also created and labeled our dataset, which consists of images taken from the recordings of the cameras installed in a parking lot. Testing results showed that our system works with high accuracy and can be applied in real-life situations. 


```{r}
pip install -r requirements.txt
```
#### Set Parking Spaces
```{r, engine='sh'}
python occupancy_detection/set_parking_spaces.py --output_path PATH_OF_OUTPUT_FILE(optional) PATH_OF_VIDEO_FILE
```
#### Detect
``````{r, engine='sh'}
python occupancy_detection/detect.py PATH_OF_VIDEO_FILE PATH_OF_PARKING_REGIONS_FILE --output_path PATH_OF_OUTPUT_FILE(optional)
```

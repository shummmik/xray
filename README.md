# Xray

Pulmonary Chest X-Ray Defect Detection

An application for detecting a Pulmonary Chest X-Ray Defect.

The graphical part of the application was created using the Streamlit — Python framework.
 
The architecture of the image processing model is U-Net.

The training of the model is carried out in jupyter notebook.

The preliminary data processing has already been carried out and saved to the data.pickle file. If it is necessary to do data preprocessing, then:
1. download data [XRAY](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels ) and put with the data directory in the root of the repository.
2. uncomment part of the code for data preprocessing

To launch the application, you must:
1. create a docker image `docker build -t xray .`
2. launch the container `docker run -p 9999:8501 -d xray`

To access the application, go to the address `localhost:9999`.
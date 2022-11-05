# A Deep Learning Architecture for Passive Microwave Precipitation Retrievals using CloudSat and GPM Data
This repository representes the development of an algorithm, called the Deep precIpitation rEtrieval alGOrithm (DIEGO) for passive microwave retrieval of precipitation. The algorithm relies on a dense and deep neural network architecture that uses coincidences of brightness temperatures (TB) from the Global Precipitation Measurment (GPM) satellite Microwave Imager (GMI) and active precipitation retrievals from the Dual-frequency Precipitation Radar (DPR) onboard the GPM as well as those from the CloudSat Cloud Profiling Radar (CPR). The algorithm first detects the precipitation occurrence and phase and then estimates its rate, while conditioning the results to some key cloud microphysical and environmental variables.

Main file: main_notebook.ipynb contains the Jyoyter notebook file explaining the use of this code, and provide the step-by-step instruction on how to use the develped algorithm for precipitation retrieval.



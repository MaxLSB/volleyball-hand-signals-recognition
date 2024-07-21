<div align="center">
  <h1>VolleyBall Referee Hand Signals Recognition </h1>
</div>

# Introduction

<div align="center">
  <img src="examples/First.gif" alt="Example" width="800"/>
</div>

# Hand Signals
<div align="center">
  <img src="assets/actions.png" alt="Interface" width="800"/>
</div>

# Installation

Use a dedicated environnement to install the librairies.

Clone repo :
```
git clone https://github.com/MaxLSB/Volleyball-Referee-Action-Recognition-Deep-Learning.git
```
Install the requirements for the app:
```
pip install -r requirements.txt
```
Download the 2 exemple videos:
```
bash downloadExamples.sh
```
To try the Detection with your webcam in real time:
```
python main.py
```
Lauching the streamlit local server (download the 2 example videos for a better experience):
```
streamlit run app.py
```




Scripts must be executed from the root folder of the project, be careful about the paths!
